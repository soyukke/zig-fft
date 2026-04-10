//! Parallel 3D FFT using threads
//!
//! Each axis FFT is parallelized across the independent slices.
//! No race conditions because each slice writes to different memory.
//!
//! Usage:
//!   - init() or initWithThreads(..., 1): Sequential mode (recommended for most cases)
//!   - initWithThreads(..., n) where n > 1: Parallel mode with n threads
//!
//! Note: Parallel mode has significant thread spawn/join overhead.
//! Only beneficial for very large grids (>128^3) with power-of-two sizes.

const std = @import("std");
const Complex = @import("complex.zig").Complex;
const fft = @import("fft.zig");
const Plan1d = fft.Plan1d;

/// Parallel 3D FFT Plan
pub const Plan3dParallel = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    plan_x: Plan1d,
    plan_y: Plan1d,
    plan_z: Plan1d,
    thread_count: usize,
    thread_scratches: [][]Complex,
    allocator: std.mem.Allocator,

    /// Initialize with sequential execution (recommended)
    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Plan3dParallel {
        return initWithThreads(allocator, nx, ny, nz, 1); // Sequential by default
    }

    /// Initialize with specified thread count
    /// - num_threads = 1: Sequential execution (no thread overhead)
    /// - num_threads > 1: Parallel execution with that many threads
    /// - num_threads = 0: Auto-detect (uses up to 8 threads)
    pub fn initWithThreads(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize, num_threads: usize) !Plan3dParallel {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        var plan_x = try Plan1d.init(allocator, nx);
        errdefer plan_x.deinit();

        var plan_y = try Plan1d.init(allocator, ny);
        errdefer plan_y.deinit();

        var plan_z = try Plan1d.init(allocator, nz);
        errdefer plan_z.deinit();

        // Determine thread count
        const thread_count = if (num_threads == 0) blk: {
            const cpu_count = std.Thread.getCpuCount() catch 4;
            break :blk @min(cpu_count, 8);
        } else num_threads;

        // Allocate per-thread scratch buffers
        const max_len = @max(ny, nz);
        const thread_scratches = try allocator.alloc([]Complex, thread_count);
        errdefer allocator.free(thread_scratches);

        var initialized_scratches: usize = 0;
        errdefer {
            for (0..initialized_scratches) |i| {
                allocator.free(thread_scratches[i]);
            }
        }

        for (0..thread_count) |i| {
            thread_scratches[i] = try allocator.alloc(Complex, max_len);
            initialized_scratches += 1;
        }

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .plan_x = plan_x,
            .plan_y = plan_y,
            .plan_z = plan_z,
            .thread_count = thread_count,
            .thread_scratches = thread_scratches,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan3dParallel) void {
        for (self.thread_scratches) |scratch| {
            self.allocator.free(scratch);
        }
        self.allocator.free(self.thread_scratches);
        self.plan_x.deinit();
        self.plan_y.deinit();
        self.plan_z.deinit();
    }

    pub fn forward(self: *Plan3dParallel, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *Plan3dParallel, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *Plan3dParallel, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // Phase 1: FFT along x-axis (parallel)
        self.parallelForX(data, inv);

        // Phase 2: FFT along y-axis (parallel)
        self.parallelForY(data, inv);

        // Phase 3: FFT along z-axis (parallel)
        self.parallelForZ(data, inv);
    }

    fn parallelForX(self: *Plan3dParallel, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;
        const total_slices = ny * nz;
        const num_threads = @min(self.thread_count, total_slices);

        // Use sequential if thread_count is 1
        if (num_threads <= 1) {
            // Single-threaded fallback
            for (0..total_slices) |idx| {
                const y = idx % ny;
                const z = idx / ny;
                const offset = nx * (y + ny * z);
                if (inv) {
                    self.plan_x.inverse(data[offset .. offset + nx]);
                } else {
                    self.plan_x.forward(data[offset .. offset + nx]);
                }
            }
            return;
        }

        const ThreadContext = struct {
            data: []Complex,
            plan_x: *const Plan1d,
            nx: usize,
            ny: usize,
            start_slice: usize,
            end_slice: usize,
            inv: bool,

            fn run(ctx: *const @This()) void {
                for (ctx.start_slice..ctx.end_slice) |idx| {
                    const y = idx % ctx.ny;
                    const z = idx / ctx.ny;
                    const offset = ctx.nx * (y + ctx.ny * z);
                    if (ctx.inv) {
                        ctx.plan_x.inverse(ctx.data[offset .. offset + ctx.nx]);
                    } else {
                        ctx.plan_x.forward(ctx.data[offset .. offset + ctx.nx]);
                    }
                }
            }
        };

        var contexts: [8]ThreadContext = undefined;
        var threads: [8]std.Thread = undefined;
        const slices_per_thread = (total_slices + num_threads - 1) / num_threads;

        for (0..num_threads) |t| {
            const start = t * slices_per_thread;
            const end = @min(start + slices_per_thread, total_slices);
            contexts[t] = .{
                .data = data,
                .plan_x = &self.plan_x,
                .nx = nx,
                .ny = ny,
                .start_slice = start,
                .end_slice = end,
                .inv = inv,
            };
            threads[t] = std.Thread.spawn(.{}, ThreadContext.run, .{&contexts[t]}) catch {
                // Fallback to sequential if spawn fails
                ThreadContext.run(&contexts[t]);
                continue;
            };
        }

        for (0..num_threads) |t| {
            threads[t].join();
        }
    }

    fn parallelForY(self: *Plan3dParallel, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;
        const num_threads = @min(self.thread_count, nz);

        // Use sequential if thread_count is 1
        if (num_threads <= 1) {
            const scratch = self.thread_scratches[0];
            for (0..nz) |z| {
                for (0..nx) |x| {
                    for (0..ny) |y| {
                        scratch[y] = data[x + nx * (y + ny * z)];
                    }
                    if (inv) {
                        self.plan_y.inverse(scratch[0..ny]);
                    } else {
                        self.plan_y.forward(scratch[0..ny]);
                    }
                    for (0..ny) |y| {
                        data[x + nx * (y + ny * z)] = scratch[y];
                    }
                }
            }
            return;
        }

        const ThreadContext = struct {
            data: []Complex,
            plan_y: *const Plan1d,
            scratch: []Complex,
            nx: usize,
            ny: usize,
            nz: usize,
            start_z: usize,
            end_z: usize,
            inv: bool,

            fn run(ctx: *const @This()) void {
                for (ctx.start_z..ctx.end_z) |z| {
                    for (0..ctx.nx) |x| {
                        for (0..ctx.ny) |y| {
                            ctx.scratch[y] = ctx.data[x + ctx.nx * (y + ctx.ny * z)];
                        }
                        if (ctx.inv) {
                            ctx.plan_y.inverse(ctx.scratch[0..ctx.ny]);
                        } else {
                            ctx.plan_y.forward(ctx.scratch[0..ctx.ny]);
                        }
                        for (0..ctx.ny) |y| {
                            ctx.data[x + ctx.nx * (y + ctx.ny * z)] = ctx.scratch[y];
                        }
                    }
                }
            }
        };

        var contexts: [8]ThreadContext = undefined;
        var threads: [8]std.Thread = undefined;
        const slices_per_thread = (nz + num_threads - 1) / num_threads;

        for (0..num_threads) |t| {
            const start = t * slices_per_thread;
            const end = @min(start + slices_per_thread, nz);
            contexts[t] = .{
                .data = data,
                .plan_y = &self.plan_y,
                .scratch = self.thread_scratches[t],
                .nx = nx,
                .ny = ny,
                .nz = nz,
                .start_z = start,
                .end_z = end,
                .inv = inv,
            };
            threads[t] = std.Thread.spawn(.{}, ThreadContext.run, .{&contexts[t]}) catch {
                ThreadContext.run(&contexts[t]);
                continue;
            };
        }

        for (0..num_threads) |t| {
            threads[t].join();
        }
    }

    fn parallelForZ(self: *Plan3dParallel, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;
        const num_threads = @min(self.thread_count, ny);

        // Use sequential if thread_count is 1
        if (num_threads <= 1) {
            const scratch = self.thread_scratches[0];
            for (0..ny) |y| {
                for (0..nx) |x| {
                    for (0..nz) |z| {
                        scratch[z] = data[x + nx * (y + ny * z)];
                    }
                    if (inv) {
                        self.plan_z.inverse(scratch[0..nz]);
                    } else {
                        self.plan_z.forward(scratch[0..nz]);
                    }
                    for (0..nz) |z| {
                        data[x + nx * (y + ny * z)] = scratch[z];
                    }
                }
            }
            return;
        }

        const ThreadContext = struct {
            data: []Complex,
            plan_z: *const Plan1d,
            scratch: []Complex,
            nx: usize,
            ny: usize,
            nz: usize,
            start_y: usize,
            end_y: usize,
            inv: bool,

            fn run(ctx: *const @This()) void {
                for (ctx.start_y..ctx.end_y) |y| {
                    for (0..ctx.nx) |x| {
                        for (0..ctx.nz) |z| {
                            ctx.scratch[z] = ctx.data[x + ctx.nx * (y + ctx.ny * z)];
                        }
                        if (ctx.inv) {
                            ctx.plan_z.inverse(ctx.scratch[0..ctx.nz]);
                        } else {
                            ctx.plan_z.forward(ctx.scratch[0..ctx.nz]);
                        }
                        for (0..ctx.nz) |z| {
                            ctx.data[x + ctx.nx * (y + ctx.ny * z)] = ctx.scratch[z];
                        }
                    }
                }
            }
        };

        var contexts: [8]ThreadContext = undefined;
        var threads: [8]std.Thread = undefined;
        const slices_per_thread = (ny + num_threads - 1) / num_threads;

        for (0..num_threads) |t| {
            const start = t * slices_per_thread;
            const end = @min(start + slices_per_thread, ny);
            contexts[t] = .{
                .data = data,
                .plan_z = &self.plan_z,
                .scratch = self.thread_scratches[t],
                .nx = nx,
                .ny = ny,
                .nz = nz,
                .start_y = start,
                .end_y = end,
                .inv = inv,
            };
            threads[t] = std.Thread.spawn(.{}, ThreadContext.run, .{&contexts[t]}) catch {
                ThreadContext.run(&contexts[t]);
                continue;
            };
        }

        for (0..num_threads) |t| {
            threads[t].join();
        }
    }
};

// ============== Tests ==============

test "Plan3dParallel roundtrip small" {
    const allocator = std.testing.allocator;

    var plan = try Plan3dParallel.initWithThreads(allocator, 4, 4, 4, 2);
    defer plan.deinit();

    var data: [64]Complex = undefined;
    for (0..64) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt(i % 7));
    }
    const original = data;

    plan.forward(&data);
    plan.inverse(&data);

    for (0..64) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-10);
    }
}

test "Plan3dParallel vs sequential" {
    const allocator = std.testing.allocator;

    var parallel_plan = try Plan3dParallel.initWithThreads(allocator, 8, 8, 8, 4);
    defer parallel_plan.deinit();

    var seq_plan = try fft.Plan3d.init(allocator, 8, 8, 8);
    defer seq_plan.deinit();

    var data_parallel: [512]Complex = undefined;
    var data_seq: [512]Complex = undefined;
    for (0..512) |i| {
        const val = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        data_parallel[i] = val;
        data_seq[i] = val;
    }

    parallel_plan.forward(&data_parallel);
    seq_plan.forward(&data_seq);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(data_parallel[i].re, data_seq[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(data_parallel[i].im, data_seq[i].im, 1e-10);
    }
}

test "Plan3dParallel large size" {
    const allocator = std.testing.allocator;

    var plan = try Plan3dParallel.init(allocator, 32, 32, 32);
    defer plan.deinit();

    const size = 32 * 32 * 32;
    const data = try allocator.alloc(Complex, size);
    defer allocator.free(data);

    for (0..size) |i| {
        data[i] = Complex.init(@floatFromInt(i % 23), @floatFromInt(i % 11));
    }

    const original = try allocator.alloc(Complex, size);
    defer allocator.free(original);
    @memcpy(original, data);

    plan.forward(data);
    plan.inverse(data);

    for (0..size) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}
