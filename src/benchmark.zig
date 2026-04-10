//! FFT Benchmark - Compare Mixed-radix vs Bluestein performance
//!
//! Run with: zig build-exe src/lib/fft/benchmark.zig -O ReleaseFast && ./benchmark

const std = @import("std");
const Complex = @import("complex.zig").Complex;
const radix2 = @import("radix2.zig");
const radix2_simd = @import("radix2_simd.zig");
const bluestein = @import("bluestein.zig");
const mixed_radix = @import("mixed_radix.zig");
const fft = @import("fft.zig");
const fft3d_parallel = @import("fft3d_parallel.zig");
const rfft = @import("rfft.zig");
const rfft3d = @import("rfft3d.zig");

const MixedRadixPlan = struct {
    n: usize,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !MixedRadixPlan {
        const scratch = try allocator.alloc(Complex, 2 * n);
        return .{ .n = n, .scratch = scratch, .allocator = allocator };
    }

    pub fn deinit(self: *MixedRadixPlan) void {
        self.allocator.free(self.scratch);
    }

    pub fn forward(self: *const MixedRadixPlan, data: []Complex) void {
        mixed_radix.mixedRadixFftNoNorm(data, self.scratch, false);
    }

    pub fn inverse(self: *const MixedRadixPlan, data: []Complex) void {
        mixed_radix.mixedRadixFftNoNorm(data, self.scratch, true);
        const scale = 1.0 / @as(f64, @floatFromInt(self.n));
        for (data) |*v| {
            v.* = Complex.scale(v.*, scale);
        }
    }
};

fn benchmark1d(allocator: std.mem.Allocator, n: usize, iterations: usize) !void {
    std.debug.print("\n=== 1D FFT Benchmark: N={d} ===\n", .{n});

    // Allocate data
    var data = try allocator.alloc(Complex, n);
    defer allocator.free(data);

    // Initialize with random-ish data
    for (0..n) |i| {
        data[i] = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    // Benchmark Bluestein
    {
        var plan = try bluestein.Plan.init(allocator, n);
        defer plan.deinit();

        const data_copy = try allocator.alloc(Complex, n);
        defer allocator.free(data_copy);

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);
            plan.forward(data_copy);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;

        std.debug.print("Bluestein:   {d:8.2} ms total, {d:8.2} µs/iter\n", .{ elapsed_ms, per_iter_us });
    }

    // Benchmark Mixed-radix (only if smooth number)
    if (mixed_radix.isSmoothNumber(n)) {
        var plan = try MixedRadixPlan.init(allocator, n);
        defer plan.deinit();

        const data_copy = try allocator.alloc(Complex, n);
        defer allocator.free(data_copy);

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);
            plan.forward(data_copy);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;

        std.debug.print("Mixed-radix: {d:8.2} ms total, {d:8.2} µs/iter\n", .{ elapsed_ms, per_iter_us });
    } else {
        std.debug.print("Mixed-radix: N/A (not a smooth number)\n", .{});
    }

    // Benchmark Radix-2 (only if power of 2)
    if (radix2.isPowerOfTwo(n)) {
        var plan = try radix2.Plan.init(allocator, n);
        defer plan.deinit();

        const data_copy = try allocator.alloc(Complex, n);
        defer allocator.free(data_copy);

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);
            plan.forward(data_copy);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;

        std.debug.print("Radix-2:     {d:8.2} ms total, {d:8.2} µs/iter\n", .{ elapsed_ms, per_iter_us });
    }

    // Benchmark Radix-2 SIMD (only if power of 2)
    if (radix2.isPowerOfTwo(n)) {
        var plan = try radix2_simd.Plan.init(allocator, n);
        defer plan.deinit();

        const data_copy = try allocator.alloc(Complex, n);
        defer allocator.free(data_copy);

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);
            plan.forward(data_copy);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;

        std.debug.print("Radix2-SIMD: {d:8.2} ms total, {d:8.2} µs/iter\n", .{ elapsed_ms, per_iter_us });
    }

    // Benchmark RFFT (only if even size)
    if (n % 2 == 0) {
        var plan = try rfft.RealPlan.init(allocator, n);
        defer plan.deinit();

        const real_data = try allocator.alloc(f64, n);
        defer allocator.free(real_data);
        for (0..n) |i| {
            real_data[i] = @floatFromInt(i % 17);
        }

        const spectrum = try allocator.alloc(Complex, n / 2 + 1);
        defer allocator.free(spectrum);

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            plan.forward(real_data, spectrum);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;

        std.debug.print("RFFT:        {d:8.2} ms total, {d:8.2} µs/iter\n", .{ elapsed_ms, per_iter_us });
    }
}

fn benchmark3d(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize, iterations: usize) !void {
    const total = nx * ny * nz;
    std.debug.print("\n=== 3D FFT Benchmark: {d}x{d}x{d} = {d} ===\n", .{ nx, ny, nz, total });

    // Allocate data
    const data = try allocator.alloc(Complex, total);
    defer allocator.free(data);

    // Initialize
    for (0..total) |i| {
        data[i] = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    const data_copy = try allocator.alloc(Complex, total);
    defer allocator.free(data_copy);

    // Scratch for y/z transforms
    const max_dim = @max(nx, @max(ny, nz));
    const scratch = try allocator.alloc(Complex, max_dim);
    defer allocator.free(scratch);

    // Benchmark with Bluestein
    {
        var plan_x = try bluestein.Plan.init(allocator, nx);
        defer plan_x.deinit();
        var plan_y = try bluestein.Plan.init(allocator, ny);
        defer plan_y.deinit();
        var plan_z = try bluestein.Plan.init(allocator, nz);
        defer plan_z.deinit();

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);

            // X-axis
            for (0..ny) |iy| {
                for (0..nz) |iz| {
                    const offset = nx * (iy + ny * iz);
                    plan_x.forward(data_copy[offset .. offset + nx]);
                }
            }

            // Y-axis
            for (0..nx) |ix| {
                for (0..nz) |iz| {
                    for (0..ny) |iy| {
                        scratch[iy] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_y.forward(scratch[0..ny]);
                    for (0..ny) |iy| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iy];
                    }
                }
            }

            // Z-axis
            for (0..nx) |ix| {
                for (0..ny) |iy| {
                    for (0..nz) |iz| {
                        scratch[iz] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_z.forward(scratch[0..nz]);
                    for (0..nz) |iz| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iz];
                    }
                }
            }
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("Bluestein:   {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    }

    // Benchmark with Mixed-radix
    if (mixed_radix.isSmoothNumber(nx) and mixed_radix.isSmoothNumber(ny) and mixed_radix.isSmoothNumber(nz)) {
        var plan_x = try MixedRadixPlan.init(allocator, nx);
        defer plan_x.deinit();
        var plan_y = try MixedRadixPlan.init(allocator, ny);
        defer plan_y.deinit();
        var plan_z = try MixedRadixPlan.init(allocator, nz);
        defer plan_z.deinit();

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);

            // X-axis
            for (0..ny) |iy| {
                for (0..nz) |iz| {
                    const offset = nx * (iy + ny * iz);
                    plan_x.forward(data_copy[offset .. offset + nx]);
                }
            }

            // Y-axis
            for (0..nx) |ix| {
                for (0..nz) |iz| {
                    for (0..ny) |iy| {
                        scratch[iy] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_y.forward(scratch[0..ny]);
                    for (0..ny) |iy| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iy];
                    }
                }
            }

            // Z-axis
            for (0..nx) |ix| {
                for (0..ny) |iy| {
                    for (0..nz) |iz| {
                        scratch[iz] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_z.forward(scratch[0..nz]);
                    for (0..nz) |iz| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iz];
                    }
                }
            }
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("Mixed-radix: {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    } else {
        std.debug.print("Mixed-radix: N/A (not all dimensions are smooth numbers)\n", .{});
    }

    // Benchmark with Radix-2 (only if all dimensions are powers of 2)
    if (radix2.isPowerOfTwo(nx) and radix2.isPowerOfTwo(ny) and radix2.isPowerOfTwo(nz)) {
        var plan_x = try radix2.Plan.init(allocator, nx);
        defer plan_x.deinit();
        var plan_y = try radix2.Plan.init(allocator, ny);
        defer plan_y.deinit();
        var plan_z = try radix2.Plan.init(allocator, nz);
        defer plan_z.deinit();

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);

            // X-axis
            for (0..ny) |iy| {
                for (0..nz) |iz| {
                    const offset = nx * (iy + ny * iz);
                    plan_x.forward(data_copy[offset .. offset + nx]);
                }
            }

            // Y-axis
            for (0..nx) |ix| {
                for (0..nz) |iz| {
                    for (0..ny) |iy| {
                        scratch[iy] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_y.forward(scratch[0..ny]);
                    for (0..ny) |iy| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iy];
                    }
                }
            }

            // Z-axis
            for (0..nx) |ix| {
                for (0..ny) |iy| {
                    for (0..nz) |iz| {
                        scratch[iz] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_z.forward(scratch[0..nz]);
                    for (0..nz) |iz| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iz];
                    }
                }
            }
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("Radix-2:     {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    }

    // Benchmark with Radix-2 SIMD (only if all dimensions are powers of 2)
    if (radix2.isPowerOfTwo(nx) and radix2.isPowerOfTwo(ny) and radix2.isPowerOfTwo(nz)) {
        var plan_x = try radix2_simd.Plan.init(allocator, nx);
        defer plan_x.deinit();
        var plan_y = try radix2_simd.Plan.init(allocator, ny);
        defer plan_y.deinit();
        var plan_z = try radix2_simd.Plan.init(allocator, nz);
        defer plan_z.deinit();

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);

            // X-axis
            for (0..ny) |iy| {
                for (0..nz) |iz| {
                    const offset = nx * (iy + ny * iz);
                    plan_x.forward(data_copy[offset .. offset + nx]);
                }
            }

            // Y-axis
            for (0..nx) |ix| {
                for (0..nz) |iz| {
                    for (0..ny) |iy| {
                        scratch[iy] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_y.forward(scratch[0..ny]);
                    for (0..ny) |iy| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iy];
                    }
                }
            }

            // Z-axis
            for (0..nx) |ix| {
                for (0..ny) |iy| {
                    for (0..nz) |iz| {
                        scratch[iz] = data_copy[ix + nx * (iy + ny * iz)];
                    }
                    plan_z.forward(scratch[0..nz]);
                    for (0..nz) |iz| {
                        data_copy[ix + nx * (iy + ny * iz)] = scratch[iz];
                    }
                }
            }
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("Radix2-SIMD: {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    }

    // Benchmark Sequential Plan3d (using fft.Plan3d)
    {
        var plan = try fft.Plan3d.init(allocator, nx, ny, nz);
        defer plan.deinit();

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);
            plan.forward(data_copy);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("Plan3d(seq): {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    }

    // Benchmark Parallel Plan3d
    {
        var plan = try fft3d_parallel.Plan3dParallel.init(allocator, nx, ny, nz);
        defer plan.deinit();

        std.debug.print("(using {d} threads)\n", .{plan.thread_count});

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            @memcpy(data_copy, data);
            plan.forward(data_copy);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("Plan3d(par): {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    }

    // Benchmark 3D RFFT (for real-valued data, only if nx is even)
    if (nx % 2 == 0) {
        var plan = try rfft3d.RealPlan3d.init(allocator, nx, ny, nz);
        defer plan.deinit();

        const real_data = try allocator.alloc(f64, total);
        defer allocator.free(real_data);
        for (0..total) |i| {
            real_data[i] = @floatFromInt(i % 17);
        }

        const complex_size = (nx / 2 + 1) * ny * nz;
        const spectrum = try allocator.alloc(Complex, complex_size);
        defer allocator.free(spectrum);

        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            plan.forward(real_data, spectrum);
        }

        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(iterations));

        std.debug.print("RFFT3d:      {d:8.2} ms total, {d:8.2} ms/iter\n", .{ elapsed_ms, per_iter_ms });
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("FFT Performance Benchmark\n", .{});
    std.debug.print("=========================\n", .{});

    // 1D benchmarks
    try benchmark1d(allocator, 24, 10000);
    try benchmark1d(allocator, 60, 10000);
    try benchmark1d(allocator, 100, 5000);
    try benchmark1d(allocator, 256, 5000);
    try benchmark1d(allocator, 1000, 1000);
    try benchmark1d(allocator, 1024, 1000);

    // 3D benchmarks (realistic DFT grid sizes)
    try benchmark3d(allocator, 24, 24, 24, 100);
    try benchmark3d(allocator, 30, 30, 30, 50);
    try benchmark3d(allocator, 32, 32, 32, 50);
    try benchmark3d(allocator, 48, 48, 48, 20);
    try benchmark3d(allocator, 64, 64, 64, 10);
    try benchmark3d(allocator, 80, 80, 80, 5);
    try benchmark3d(allocator, 96, 96, 96, 5);

    std.debug.print("\nBenchmark complete.\n", .{});
}
