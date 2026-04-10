//! Parallel 3D FFT implementation optimized for N=24 grids
//!
//! Uses comptime-precomputed twiddle factors for maximum performance.
//! Specialized for 24×24×24 grids common in DFT calculations.

const std = @import("std");
pub const Complex = @import("complex.zig").Complex;
const fft24_comptime = @import("fft24_comptime.zig");

/// Thread-local workspace for parallel FFT-24
const ThreadWorkspace = struct {
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_len: usize) !ThreadWorkspace {
        const scratch = try allocator.alloc(Complex, max_len);
        return .{
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadWorkspace) void {
        if (self.scratch.len > 0) self.allocator.free(self.scratch);
    }
};

/// Task type for thread pool
const TaskType = enum {
    none,
    fft_x,
    fft_y,
    fft_z,
    shutdown,
};

/// Shared state for thread pool synchronization
const ThreadPoolState = struct {
    task: TaskType,
    data: ?[]Complex,
    inverse: bool,
    nx: usize,
    ny: usize,
    nz: usize,
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
            .data = null,
            .inverse = false,
            .nx = nx,
            .ny = ny,
            .nz = nz,
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

/// Parallel 3D FFT Plan optimized for 24×24×24 grids
pub const ParallelPlan3d24 = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    num_threads: usize,
    workspaces: []ThreadWorkspace,
    threads: []std.Thread,
    state: *ThreadPoolState,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !ParallelPlan3d24 {
        return initWithThreads(allocator, nx, ny, nz, 0);
    }

    pub fn initWithThreads(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize, num_threads_hint: usize) !ParallelPlan3d24 {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;
        // Only support 24×24×24 grids
        if (nx != 24 or ny != 24 or nz != 24) return error.UnsupportedSize;

        const cpu_count = std.Thread.getCpuCount() catch 4;
        const num_threads = if (num_threads_hint == 0) @min(cpu_count, 16) else num_threads_hint;

        const workspaces = try allocator.alloc(ThreadWorkspace, num_threads);
        errdefer allocator.free(workspaces);

        var init_count: usize = 0;
        errdefer {
            for (0..init_count) |i| {
                workspaces[i].deinit();
            }
        }

        const max_len = @max(nx, @max(ny, nz));
        for (0..num_threads) |i| {
            workspaces[i] = try ThreadWorkspace.init(allocator, max_len);
            init_count += 1;
        }

        const state = try allocator.create(ThreadPoolState);
        errdefer allocator.destroy(state);
        state.* = ThreadPoolState.init(nx, ny, nz, num_threads);

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

        for (0..num_threads) |i| {
            threads[i] = try std.Thread.spawn(.{}, workerThread, .{ state, &workspaces[i] });
            spawned += 1;
        }

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .num_threads = num_threads,
            .workspaces = workspaces,
            .threads = threads,
            .state = state,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ParallelPlan3d24) void {
        self.state.mutex.lock();
        self.state.task_generation += 1;
        self.state.task = .shutdown;
        self.state.work_available.broadcast();
        self.state.mutex.unlock();

        for (self.threads) |t| {
            t.join();
        }

        for (self.workspaces) |*ws| {
            ws.deinit();
        }
        self.allocator.free(self.workspaces);
        self.allocator.free(self.threads);
        self.allocator.destroy(self.state);
    }

    pub fn forward(self: *ParallelPlan3d24, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *ParallelPlan3d24, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *ParallelPlan3d24, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // FFT along x-axis (parallel over y*z)
        self.dispatchTask(.fft_x, data, inv, ny * nz);

        // FFT along y-axis (parallel over x*z)
        self.dispatchTask(.fft_y, data, inv, nx * nz);

        // FFT along z-axis (parallel over x*y)
        self.dispatchTask(.fft_z, data, inv, nx * ny);
    }

    fn dispatchTask(self: *ParallelPlan3d24, task: TaskType, data: []Complex, inv: bool, total_work: usize) void {
        self.state.mutex.lock();

        self.state.task_generation += 1;
        self.state.task = task;
        self.state.data = data;
        self.state.inverse = inv;
        self.state.total_work = total_work;
        self.state.next_work_item.store(0, .seq_cst);
        self.state.barrier_count.store(0, .seq_cst);

        self.state.work_available.broadcast();
        self.state.mutex.unlock();

        self.state.mutex.lock();
        while (self.state.barrier_count.load(.seq_cst) < self.num_threads) {
            self.state.work_done.wait(&self.state.mutex);
        }

        self.state.task = .none;
        self.state.mutex.unlock();
    }

    fn workerThread(state: *ThreadPoolState, ws: *ThreadWorkspace) void {
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
            const data = state.data;
            const inv = state.inverse;
            last_generation = state.task_generation;
            state.mutex.unlock();

            if (task == .shutdown) {
                return;
            }

            if (data) |d| {
                while (true) {
                    const item = state.next_work_item.fetchAdd(1, .seq_cst);
                    if (item >= state.total_work) break;

                    switch (task) {
                        .fft_x => processXAxis(state, d, inv, item),
                        .fft_y => processYAxis(state, ws, d, inv, item),
                        .fft_z => processZAxis(state, ws, d, inv, item),
                        else => {},
                    }
                }
            }

            const count = state.barrier_count.fetchAdd(1, .seq_cst) + 1;
            if (count == state.num_threads) {
                state.mutex.lock();
                state.work_done.signal();
                state.mutex.unlock();
            }
        }
    }

    fn processXAxis(state: *ThreadPoolState, data: []Complex, inv: bool, item: usize) void {
        const nx = state.nx;
        const ny = state.ny;
        const y = item % ny;
        const z = item / ny;
        const offset = nx * (y + ny * z);

        // Use comptime-optimized FFT-24
        fft24_comptime.fft24(data[offset .. offset + nx], inv);
    }

    fn processYAxis(state: *ThreadPoolState, ws: *ThreadWorkspace, data: []Complex, inv: bool, item: usize) void {
        const nx = state.nx;
        const ny = state.ny;
        const x = item % nx;
        const z = item / nx;

        // Gather
        for (0..ny) |j| {
            ws.scratch[j] = data[x + nx * (j + ny * z)];
        }

        // Use comptime-optimized FFT-24
        fft24_comptime.fft24(ws.scratch[0..ny], inv);

        // Scatter
        for (0..ny) |j| {
            data[x + nx * (j + ny * z)] = ws.scratch[j];
        }
    }

    fn processZAxis(state: *ThreadPoolState, ws: *ThreadWorkspace, data: []Complex, inv: bool, item: usize) void {
        const nx = state.nx;
        const ny = state.ny;
        const nz = state.nz;
        const x = item % nx;
        const y = item / nx;

        // Gather
        for (0..nz) |k| {
            ws.scratch[k] = data[x + nx * (y + ny * k)];
        }

        // Use comptime-optimized FFT-24
        fft24_comptime.fft24(ws.scratch[0..nz], inv);

        // Scatter
        for (0..nz) |k| {
            data[x + nx * (y + ny * k)] = ws.scratch[k];
        }
    }
};

// ============== Tests ==============

test "ParallelPlan3d24 roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try ParallelPlan3d24.initWithThreads(allocator, 24, 24, 24, 4);
    defer plan.deinit();

    const total = 24 * 24 * 24;
    var data: [total]Complex = undefined;
    var original: [total]Complex = undefined;
    for (0..total) |i| {
        data[i] = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        original[i] = data[i];
    }

    plan.forward(&data);
    plan.inverse(&data);

    for (0..total) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "ParallelPlan3d24 matches parallel_fft" {
    const allocator = std.testing.allocator;
    const parallel_fft = @import("parallel_fft.zig");

    var plan24 = try ParallelPlan3d24.initWithThreads(allocator, 24, 24, 24, 4);
    defer plan24.deinit();

    var plan_ref = try parallel_fft.ParallelPlan3d.initWithThreads(allocator, 24, 24, 24, 4);
    defer plan_ref.deinit();

    const total = 24 * 24 * 24;
    var data24: [total]Complex = undefined;
    var data_ref: [total]Complex = undefined;
    for (0..total) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        data24[i] = val;
        data_ref[i] = val;
    }

    plan24.forward(&data24);
    plan_ref.forward(&data_ref);

    for (0..total) |i| {
        try std.testing.expectApproxEqAbs(data24[i].re, data_ref[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data24[i].im, data_ref[i].im, 1e-9);
    }
}
