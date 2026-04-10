//! Transpose-based Parallel 3D FFT implementation
//!
//! Uses transpose operations to ensure all FFTs operate on contiguous memory.
//! Pattern: FFT(x) -> Transpose(xyz->yxz) -> FFT(y) -> Transpose(yxz->zxy) -> FFT(z) -> Transpose(zxy->xyz)
//!
//! This maximizes cache efficiency by avoiding strided memory access.

const std = @import("std");
pub const Complex = @import("complex.zig").Complex;
const Plan1d = @import("fft.zig").Plan1d;

/// Thread-local workspace for parallel FFT
const ThreadWorkspace = struct {
    plan: Plan1d, // Single plan for the current axis size
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !ThreadWorkspace {
        const plan = try Plan1d.init(allocator, n);
        return .{
            .plan = plan,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadWorkspace) void {
        self.plan.deinit();
    }
};

/// Task type for thread pool
const TaskType = enum {
    none,
    fft_x, // FFT along x-axis
    fft_y, // FFT along y-axis
    fft_z, // FFT along z-axis
    transpose_xyz_yxz,
    transpose_yxz_zxy,
    transpose_zxy_xyz,
    shutdown,
};

/// Shared state for thread pool synchronization
const ThreadPoolState = struct {
    task: TaskType,
    src: ?[]Complex,
    dst: ?[]Complex,
    inverse: bool,
    nx: usize,
    ny: usize,
    nz: usize,
    axis_size: usize, // Current FFT axis size
    total_work: usize,
    next_work_item: std.atomic.Value(usize),
    task_generation: usize,
    barrier_count: std.atomic.Value(usize),
    num_threads: usize,
    mutex: std.Thread.Mutex,
    work_available: std.Thread.Condition,
    work_done: std.Thread.Condition,

    fn init(nx: usize, ny: usize, nz: usize, num_threads: usize) ThreadPoolState {
        return .{
            .task = .none,
            .src = null,
            .dst = null,
            .inverse = false,
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .axis_size = 0,
            .total_work = 0,
            .next_work_item = std.atomic.Value(usize).init(0),
            .task_generation = 0,
            .barrier_count = std.atomic.Value(usize).init(0),
            .num_threads = num_threads,
            .mutex = .{},
            .work_available = .{},
            .work_done = .{},
        };
    }
};

/// Transpose-based Parallel 3D FFT Plan
pub const TransposePlan3d = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    num_threads: usize,
    workspaces_x: []ThreadWorkspace,
    workspaces_y: []ThreadWorkspace,
    workspaces_z: []ThreadWorkspace,
    buffer: []Complex, // Temporary buffer for transpose
    threads: []std.Thread,
    state: *ThreadPoolState,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !TransposePlan3d {
        return initWithThreads(allocator, nx, ny, nz, 0);
    }

    pub fn initWithThreads(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize, num_threads_hint: usize) !TransposePlan3d {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        const cpu_count = std.Thread.getCpuCount() catch 4;
        const num_threads = if (num_threads_hint == 0) @min(cpu_count, 16) else num_threads_hint;

        // Allocate workspaces for each axis
        const workspaces_x = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(workspaces_x);

        const workspaces_y = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(workspaces_y);

        const workspaces_z = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(workspaces_z);

        var init_x: usize = 0;
        var init_y: usize = 0;
        var init_z: usize = 0;

        errdefer {
            for (0..init_x) |i| workspaces_x[i].deinit();
            for (0..init_y) |i| workspaces_y[i].deinit();
            for (0..init_z) |i| workspaces_z[i].deinit();
        }

        for (0..num_threads) |i| {
            workspaces_x[i] = try ThreadWorkspace.init(allocator, nx);
            init_x += 1;
        }
        for (0..num_threads) |i| {
            workspaces_y[i] = try ThreadWorkspace.init(allocator, ny);
            init_y += 1;
        }
        for (0..num_threads) |i| {
            workspaces_z[i] = try ThreadWorkspace.init(allocator, nz);
            init_z += 1;
        }

        // Allocate transpose buffer
        const buffer = try allocator.alloc(Complex, nx * ny * nz);
        errdefer allocator.free(buffer);

        // Allocate shared state
        const state = try allocator.create(ThreadPoolState);
        errdefer allocator.destroy(state);
        state.* = ThreadPoolState.init(nx, ny, nz, num_threads);

        // Allocate thread handles
        const threads = try allocator.alloc(std.Thread, num_threads);
        errdefer allocator.free(threads);

        var spawned: usize = 0;
        errdefer {
            state.mutex.lock();
            state.task_generation += 1;
            state.task = .shutdown;
            state.work_available.broadcast();
            state.mutex.unlock();
            for (0..spawned) |i| {
                threads[i].join();
            }
        }

        // Spawn worker threads
        for (0..num_threads) |i| {
            threads[i] = try std.Thread.spawn(.{}, workerThread, .{
                state,
                &workspaces_x[i],
                &workspaces_y[i],
                &workspaces_z[i],
            });
            spawned += 1;
        }

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .num_threads = num_threads,
            .workspaces_x = workspaces_x,
            .workspaces_y = workspaces_y,
            .workspaces_z = workspaces_z,
            .buffer = buffer,
            .threads = threads,
            .state = state,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TransposePlan3d) void {
        // Signal shutdown
        self.state.mutex.lock();
        self.state.task_generation += 1;
        self.state.task = .shutdown;
        self.state.work_available.broadcast();
        self.state.mutex.unlock();

        // Wait for all threads to finish
        for (self.threads) |t| {
            t.join();
        }

        // Clean up
        for (self.workspaces_x) |*ws| ws.deinit();
        for (self.workspaces_y) |*ws| ws.deinit();
        for (self.workspaces_z) |*ws| ws.deinit();
        self.allocator.free(self.workspaces_x);
        self.allocator.free(self.workspaces_y);
        self.allocator.free(self.workspaces_z);
        self.allocator.free(self.buffer);
        self.allocator.free(self.threads);
        self.allocator.destroy(self.state);
    }

    pub fn forward(self: *TransposePlan3d, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *TransposePlan3d, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *TransposePlan3d, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // Step 1: FFT along x-axis (data is xyz, x is contiguous)
        self.dispatchFft(.fft_x, data, nx, ny * nz, inv);

        // Step 2: Transpose xyz -> yxz
        self.dispatchTranspose(.transpose_xyz_yxz, data, self.buffer);

        // Step 3: FFT along y-axis (buffer is yxz, y is contiguous)
        self.dispatchFft(.fft_y, self.buffer, ny, nx * nz, inv);

        // Step 4: Transpose yxz -> zxy
        self.dispatchTranspose(.transpose_yxz_zxy, self.buffer, data);

        // Step 5: FFT along z-axis (data is zxy, z is contiguous)
        self.dispatchFft(.fft_z, data, nz, nx * ny, inv);

        // Step 6: Transpose zxy -> xyz
        self.dispatchTranspose(.transpose_zxy_xyz, data, self.buffer);

        // Copy result back to data
        @memcpy(data, self.buffer);
    }

    fn dispatchFft(self: *TransposePlan3d, task: TaskType, data: []Complex, axis_size: usize, num_ffts: usize, inv: bool) void {
        self.state.mutex.lock();

        self.state.task_generation += 1;
        self.state.task = task;
        self.state.src = data;
        self.state.dst = null;
        self.state.inverse = inv;
        self.state.axis_size = axis_size;
        self.state.total_work = num_ffts;
        self.state.next_work_item.store(0, .seq_cst);
        self.state.barrier_count.store(0, .seq_cst);

        self.state.work_available.broadcast();
        self.state.mutex.unlock();

        // Wait for completion
        self.state.mutex.lock();
        while (self.state.barrier_count.load(.seq_cst) < self.num_threads) {
            self.state.work_done.wait(&self.state.mutex);
        }
        self.state.task = .none;
        self.state.mutex.unlock();
    }

    fn dispatchTranspose(self: *TransposePlan3d, task: TaskType, src: []Complex, dst: []Complex) void {
        self.state.mutex.lock();

        self.state.task_generation += 1;
        self.state.task = task;
        self.state.src = src;
        self.state.dst = dst;
        self.state.total_work = self.computeTransposeWork(task);
        self.state.next_work_item.store(0, .seq_cst);
        self.state.barrier_count.store(0, .seq_cst);

        self.state.work_available.broadcast();
        self.state.mutex.unlock();

        // Wait for completion
        self.state.mutex.lock();
        while (self.state.barrier_count.load(.seq_cst) < self.num_threads) {
            self.state.work_done.wait(&self.state.mutex);
        }
        self.state.task = .none;
        self.state.mutex.unlock();
    }

    fn computeTransposeWork(self: *TransposePlan3d, task: TaskType) usize {
        return switch (task) {
            .transpose_xyz_yxz => self.ny * self.nz, // parallelize over y*z
            .transpose_yxz_zxy => self.nz * self.nx, // parallelize over z*x
            .transpose_zxy_xyz => self.nx * self.ny, // parallelize over x*y
            else => 0,
        };
    }

    fn workerThread(
        state: *ThreadPoolState,
        ws_x: *ThreadWorkspace,
        ws_y: *ThreadWorkspace,
        ws_z: *ThreadWorkspace,
    ) void {
        var last_generation: usize = 0;

        while (true) {
            state.mutex.lock();
            while (state.task == .none or state.task_generation == last_generation) {
                if (state.task == .shutdown) {
                    state.mutex.unlock();
                    return;
                }
                state.work_available.wait(&state.mutex);
            }

            const task = state.task;
            const src = state.src;
            const dst = state.dst;
            const inv = state.inverse;
            const axis_size = state.axis_size;
            last_generation = state.task_generation;
            state.mutex.unlock();

            if (task == .shutdown) return;

            // Process work items
            while (true) {
                const item = state.next_work_item.fetchAdd(1, .seq_cst);
                if (item >= state.total_work) break;

                switch (task) {
                    .fft_x => {
                        if (src) |s| {
                            processFftAxis(s, axis_size, item, inv, ws_x);
                        }
                    },
                    .fft_y => {
                        if (src) |s| {
                            processFftAxis(s, axis_size, item, inv, ws_y);
                        }
                    },
                    .fft_z => {
                        if (src) |s| {
                            processFftAxis(s, axis_size, item, inv, ws_z);
                        }
                    },
                    .transpose_xyz_yxz => {
                        if (src != null and dst != null) {
                            transposeXyzToYxz(state, src.?, dst.?, item);
                        }
                    },
                    .transpose_yxz_zxy => {
                        if (src != null and dst != null) {
                            transposeYxzToZxy(state, src.?, dst.?, item);
                        }
                    },
                    .transpose_zxy_xyz => {
                        if (src != null and dst != null) {
                            transposeZxyToXyz(state, src.?, dst.?, item);
                        }
                    },
                    else => {},
                }
            }

            // Barrier
            const count = state.barrier_count.fetchAdd(1, .seq_cst) + 1;
            if (count == state.num_threads) {
                state.mutex.lock();
                state.work_done.signal();
                state.mutex.unlock();
            }
        }
    }

    fn processFftAxis(data: []Complex, axis_size: usize, item: usize, inv: bool, ws: *ThreadWorkspace) void {
        const offset = item * axis_size;
        if (inv) {
            ws.plan.inverse(data[offset .. offset + axis_size]);
        } else {
            ws.plan.forward(data[offset .. offset + axis_size]);
        }
    }

    // xyz[x + nx*(y + ny*z)] -> yxz[y + ny*(x + nx*z)]
    fn transposeXyzToYxz(state: *ThreadPoolState, src: []Complex, dst: []Complex, item: usize) void {
        const nx = state.nx;
        const ny = state.ny;
        const y = item % ny;
        const z = item / ny;

        for (0..nx) |x| {
            const src_idx = x + nx * (y + ny * z);
            const dst_idx = y + ny * (x + nx * z);
            dst[dst_idx] = src[src_idx];
        }
    }

    // yxz[y + ny*(x + nx*z)] -> zxy[z + nz*(x + nx*y)]
    fn transposeYxzToZxy(state: *ThreadPoolState, src: []Complex, dst: []Complex, item: usize) void {
        const nx = state.nx;
        const ny = state.ny;
        const nz = state.nz;
        const z = item % nz;
        const x = item / nz;

        for (0..ny) |y| {
            const src_idx = y + ny * (x + nx * z);
            const dst_idx = z + nz * (x + nx * y);
            dst[dst_idx] = src[src_idx];
        }
    }

    // zxy[z + nz*(x + nx*y)] -> xyz[x + nx*(y + ny*z)]
    fn transposeZxyToXyz(state: *ThreadPoolState, src: []Complex, dst: []Complex, item: usize) void {
        const nx = state.nx;
        const ny = state.ny;
        const nz = state.nz;
        const x = item % nx;
        const y = item / nx;

        for (0..nz) |z| {
            const src_idx = z + nz * (x + nx * y);
            const dst_idx = x + nx * (y + ny * z);
            dst[dst_idx] = src[src_idx];
        }
    }
};

// ============== Tests ==============

test "TransposePlan3d roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try TransposePlan3d.initWithThreads(allocator, 8, 8, 8, 4);
    defer plan.deinit();

    var data: [512]Complex = undefined;
    var original: [512]Complex = undefined;
    for (0..512) |i| {
        data[i] = Complex.init(@floatFromInt(i), 0);
        original[i] = data[i];
    }

    plan.forward(&data);
    plan.inverse(&data);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "TransposePlan3d matches sequential" {
    const allocator = std.testing.allocator;
    const Plan3d = @import("fft.zig").Plan3d;

    var trans_plan = try TransposePlan3d.initWithThreads(allocator, 8, 8, 8, 4);
    defer trans_plan.deinit();

    var seq_plan = try Plan3d.init(allocator, 8, 8, 8);
    defer seq_plan.deinit();

    var trans_data: [512]Complex = undefined;
    var seq_data: [512]Complex = undefined;
    for (0..512) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        trans_data[i] = val;
        seq_data[i] = val;
    }

    trans_plan.forward(&trans_data);
    seq_plan.forward(&seq_data);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(trans_data[i].re, seq_data[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(trans_data[i].im, seq_data[i].im, 1e-9);
    }
}

test "TransposePlan3d non-cubic" {
    const allocator = std.testing.allocator;
    const Plan3d = @import("fft.zig").Plan3d;

    // Test with 24x24x24 (non-power-of-2)
    var trans_plan = try TransposePlan3d.initWithThreads(allocator, 24, 24, 24, 4);
    defer trans_plan.deinit();

    var seq_plan = try Plan3d.init(allocator, 24, 24, 24);
    defer seq_plan.deinit();

    const size = 24 * 24 * 24;
    var trans_data = try allocator.alloc(Complex, size);
    defer allocator.free(trans_data);
    var seq_data = try allocator.alloc(Complex, size);
    defer allocator.free(seq_data);

    for (0..size) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        trans_data[i] = val;
        seq_data[i] = val;
    }

    trans_plan.forward(trans_data);
    seq_plan.forward(seq_data);

    for (0..size) |i| {
        try std.testing.expectApproxEqAbs(trans_data[i].re, seq_data[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(trans_data[i].im, seq_data[i].im, 1e-9);
    }
}
