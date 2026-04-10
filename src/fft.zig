//! FFT Library - Arbitrary size FFT support
//!
//! Supports any size N, automatically selecting the best algorithm:
//! - Power of 2: Radix-2 Cooley-Tukey (fastest)
//! - Smooth numbers (2^a × 3^b × 5^c): Mixed-radix Cooley-Tukey
//! - Other sizes: Bluestein's algorithm
//!
//! Usage:
//! ```zig
//! const fft = @import("lib/fft/fft.zig");
//!
//! var plan = try fft.Plan1d.init(allocator, 24);
//! defer plan.deinit();
//!
//! plan.forward(data);
//! plan.inverse(data);
//! ```

const std = @import("std");
const builtin = @import("builtin");
pub const Complex = @import("complex.zig").Complex;
const radix2 = @import("radix2_simd.zig"); // SIMD-optimized version
const radix2_scalar = @import("radix2.zig"); // Scalar version for benchmarking
pub const rfft3d = @import("rfft3d.zig"); // Real FFT 3D for benchmarking
const bluestein = @import("bluestein.zig");
pub const mixed_radix = @import("mixed_radix.zig"); // Mixed-radix FFT
pub const vdsp_fft = @import("vdsp_fft.zig"); // vDSP FFT for macOS Accelerate
pub const parallel_fft = @import("parallel_fft.zig"); // Parallel 3D FFT with threading
pub const parallel_fft_transpose = @import("parallel_fft_transpose.zig"); // Transpose-based parallel FFT
pub const fftw_fft = @import("fftw_fft.zig"); // FFTW3 FFT (requires linking)
pub const metal_fft = @import("metal_fft.zig"); // Metal GPU FFT (macOS only)
pub const fft24_comptime = @import("fft24_comptime.zig"); // Comptime-optimized FFT-24
pub const parallel_fft24 = @import("parallel_fft24.zig"); // Parallel FFT optimized for 24×24×24

/// Check if value is power of two.
pub fn isPowerOfTwo(n: usize) bool {
    return radix2.isPowerOfTwo(n);
}

/// Check if n is a smooth number (composed only of factors 2, 3, 5).
pub fn isSmoothNumber(n: usize) bool {
    return mixed_radix.isSmoothNumber(n);
}

/// Mixed-radix plan wrapper (SIMD version).
const MixedRadixPlan = struct {
    n: usize,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !MixedRadixPlan {
        const scratch = try allocator.alloc(Complex, 2 * n);
        return .{
            .n = n,
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MixedRadixPlan) void {
        self.allocator.free(self.scratch);
    }

    pub fn forward(self: *const MixedRadixPlan, data: []Complex) void {
        if (data.len != self.n) return;
        mixed_radix.mixedRadixFftNoNorm(data, self.scratch, false);
    }

    pub fn inverse(self: *const MixedRadixPlan, data: []Complex) void {
        if (data.len != self.n) return;
        mixed_radix.mixedRadixFftNoNorm(data, self.scratch, true);
        // Apply normalization
        const scale = 1.0 / @as(f64, @floatFromInt(self.n));
        for (data) |*v| {
            v.* = Complex.scale(v.*, scale);
        }
    }
};

/// Mixed-radix plan wrapper (scalar version for benchmarking).
const MixedRadixPlanScalar = struct {
    n: usize,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !MixedRadixPlanScalar {
        const scratch = try allocator.alloc(Complex, 2 * n);
        return .{
            .n = n,
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MixedRadixPlanScalar) void {
        self.allocator.free(self.scratch);
    }

    pub fn forward(self: *const MixedRadixPlanScalar, data: []Complex) void {
        if (data.len != self.n) return;
        mixed_radix.mixedRadixFftNoNormScalar(data, self.scratch, false);
    }

    pub fn inverse(self: *const MixedRadixPlanScalar, data: []Complex) void {
        if (data.len != self.n) return;
        mixed_radix.mixedRadixFftNoNormScalar(data, self.scratch, true);
        const scale = 1.0 / @as(f64, @floatFromInt(self.n));
        for (data) |*v| {
            v.* = Complex.scale(v.*, scale);
        }
    }
};

/// 1D FFT Plan - automatically selects optimal algorithm.
pub const Plan1d = struct {
    n: usize,
    algorithm: Algorithm,
    allocator: std.mem.Allocator,

    const Algorithm = union(enum) {
        radix2: radix2.Plan,
        mixed_radix: MixedRadixPlan,
        bluestein: bluestein.Plan,
    };

    pub fn init(allocator: std.mem.Allocator, n: usize) !Plan1d {
        if (n == 0) return error.InvalidSize;

        if (isPowerOfTwo(n)) {
            const plan = try radix2.Plan.init(allocator, n);
            return .{
                .n = n,
                .algorithm = .{ .radix2 = plan },
                .allocator = allocator,
            };
        } else if (isSmoothNumber(n)) {
            const plan = try MixedRadixPlan.init(allocator, n);
            return .{
                .n = n,
                .algorithm = .{ .mixed_radix = plan },
                .allocator = allocator,
            };
        } else {
            const plan = try bluestein.Plan.init(allocator, n);
            return .{
                .n = n,
                .algorithm = .{ .bluestein = plan },
                .allocator = allocator,
            };
        }
    }

    pub fn deinit(self: *Plan1d) void {
        switch (self.algorithm) {
            .radix2 => |*p| p.deinit(),
            .mixed_radix => |*p| p.deinit(),
            .bluestein => |*p| p.deinit(),
        }
    }

    pub fn forward(self: *const Plan1d, data: []Complex) void {
        if (data.len != self.n) return;
        switch (self.algorithm) {
            .radix2 => |*p| p.forward(data),
            .mixed_radix => |*p| p.forward(data),
            .bluestein => |*p| p.forward(data),
        }
    }

    pub fn inverse(self: *const Plan1d, data: []Complex) void {
        if (data.len != self.n) return;
        switch (self.algorithm) {
            .radix2 => |*p| p.inverse(data),
            .mixed_radix => |*p| p.inverse(data),
            .bluestein => |*p| p.inverse(data),
        }
    }
};

/// 1D FFT Plan using scalar algorithms (for benchmarking).
pub const Plan1dScalar = struct {
    n: usize,
    algorithm: AlgorithmScalar,
    allocator: std.mem.Allocator,

    const AlgorithmScalar = union(enum) {
        radix2: radix2_scalar.Plan, // Scalar radix-2
        mixed_radix: MixedRadixPlanScalar,
        bluestein: bluestein.Plan,
    };

    pub fn init(allocator: std.mem.Allocator, n: usize) !Plan1dScalar {
        if (n == 0) return error.InvalidSize;

        if (isPowerOfTwo(n)) {
            const plan = try radix2_scalar.Plan.init(allocator, n);
            return .{
                .n = n,
                .algorithm = .{ .radix2 = plan },
                .allocator = allocator,
            };
        } else if (isSmoothNumber(n)) {
            const plan = try MixedRadixPlanScalar.init(allocator, n);
            return .{
                .n = n,
                .algorithm = .{ .mixed_radix = plan },
                .allocator = allocator,
            };
        } else {
            const plan = try bluestein.Plan.init(allocator, n);
            return .{
                .n = n,
                .algorithm = .{ .bluestein = plan },
                .allocator = allocator,
            };
        }
    }

    pub fn deinit(self: *Plan1dScalar) void {
        switch (self.algorithm) {
            .radix2 => |*p| p.deinit(),
            .mixed_radix => |*p| p.deinit(),
            .bluestein => |*p| p.deinit(),
        }
    }

    pub fn forward(self: *const Plan1dScalar, data: []Complex) void {
        if (data.len != self.n) return;
        switch (self.algorithm) {
            .radix2 => |*p| p.forward(data),
            .mixed_radix => |*p| p.forward(data),
            .bluestein => |*p| p.forward(data),
        }
    }

    pub fn inverse(self: *const Plan1dScalar, data: []Complex) void {
        if (data.len != self.n) return;
        switch (self.algorithm) {
            .radix2 => |*p| p.inverse(data),
            .mixed_radix => |*p| p.inverse(data),
            .bluestein => |*p| p.inverse(data),
        }
    }
};

/// 3D FFT Plan using scalar mixed-radix (for benchmarking).
pub const Plan3dScalar = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    plan_x: Plan1dScalar,
    plan_y: Plan1dScalar,
    plan_z: Plan1dScalar,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Plan3dScalar {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        var plan_x = try Plan1dScalar.init(allocator, nx);
        errdefer plan_x.deinit();

        var plan_y = try Plan1dScalar.init(allocator, ny);
        errdefer plan_y.deinit();

        var plan_z = try Plan1dScalar.init(allocator, nz);
        errdefer plan_z.deinit();

        const max_len = @max(nx, @max(ny, nz));
        const scratch = try allocator.alloc(Complex, max_len);
        errdefer allocator.free(scratch);

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .plan_x = plan_x,
            .plan_y = plan_y,
            .plan_z = plan_z,
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan3dScalar) void {
        self.plan_x.deinit();
        self.plan_y.deinit();
        self.plan_z.deinit();
        if (self.scratch.len > 0) self.allocator.free(self.scratch);
    }

    pub fn forward(self: *Plan3dScalar, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *Plan3dScalar, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *Plan3dScalar, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // FFT along x-axis
        for (0..ny) |y| {
            for (0..nz) |z| {
                const offset = nx * (y + ny * z);
                if (inv) {
                    self.plan_x.inverse(data[offset .. offset + nx]);
                } else {
                    self.plan_x.forward(data[offset .. offset + nx]);
                }
            }
        }

        // FFT along y-axis
        const buffer = self.scratch;
        for (0..nx) |x| {
            for (0..nz) |z| {
                for (0..ny) |j| {
                    buffer[j] = data[x + nx * (j + ny * z)];
                }
                if (inv) {
                    self.plan_y.inverse(buffer[0..ny]);
                } else {
                    self.plan_y.forward(buffer[0..ny]);
                }
                for (0..ny) |j| {
                    data[x + nx * (j + ny * z)] = buffer[j];
                }
            }
        }

        // FFT along z-axis
        for (0..nx) |x| {
            for (0..ny) |y| {
                for (0..nz) |k| {
                    buffer[k] = data[x + nx * (y + ny * k)];
                }
                if (inv) {
                    self.plan_z.inverse(buffer[0..nz]);
                } else {
                    self.plan_z.forward(buffer[0..nz]);
                }
                for (0..nz) |k| {
                    data[x + nx * (y + ny * k)] = buffer[k];
                }
            }
        }
    }
};

/// 3D FFT Plan - applies 1D FFT along each axis.
pub const Plan3d = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    plan_x: Plan1d,
    plan_y: Plan1d,
    plan_z: Plan1d,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !Plan3d {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        var plan_x = try Plan1d.init(allocator, nx);
        errdefer plan_x.deinit();

        var plan_y = try Plan1d.init(allocator, ny);
        errdefer plan_y.deinit();

        var plan_z = try Plan1d.init(allocator, nz);
        errdefer plan_z.deinit();

        const max_len = @max(nx, @max(ny, nz));
        const scratch = try allocator.alloc(Complex, max_len);
        errdefer allocator.free(scratch);

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .plan_x = plan_x,
            .plan_y = plan_y,
            .plan_z = plan_z,
            .scratch = scratch,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan3d) void {
        self.plan_x.deinit();
        self.plan_y.deinit();
        self.plan_z.deinit();
        if (self.scratch.len > 0) self.allocator.free(self.scratch);
    }

    pub fn forward(self: *Plan3d, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *Plan3d, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *Plan3d, data: []Complex, inv: bool) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;

        if (data.len != nx * ny * nz) return;

        // FFT along x-axis
        var y: usize = 0;
        while (y < ny) : (y += 1) {
            var z: usize = 0;
            while (z < nz) : (z += 1) {
                const offset = nx * (y + ny * z);
                if (inv) {
                    self.plan_x.inverse(data[offset .. offset + nx]);
                } else {
                    self.plan_x.forward(data[offset .. offset + nx]);
                }
            }
        }

        // FFT along y-axis
        const buffer = self.scratch;
        var x: usize = 0;
        while (x < nx) : (x += 1) {
            var z: usize = 0;
            while (z < nz) : (z += 1) {
                var j: usize = 0;
                while (j < ny) : (j += 1) {
                    buffer[j] = data[x + nx * (j + ny * z)];
                }
                if (inv) {
                    self.plan_y.inverse(buffer[0..ny]);
                } else {
                    self.plan_y.forward(buffer[0..ny]);
                }
                j = 0;
                while (j < ny) : (j += 1) {
                    data[x + nx * (j + ny * z)] = buffer[j];
                }
            }
        }

        // FFT along z-axis
        x = 0;
        while (x < nx) : (x += 1) {
            var y2: usize = 0;
            while (y2 < ny) : (y2 += 1) {
                var k: usize = 0;
                while (k < nz) : (k += 1) {
                    buffer[k] = data[x + nx * (y2 + ny * k)];
                }
                if (inv) {
                    self.plan_z.inverse(buffer[0..nz]);
                } else {
                    self.plan_z.forward(buffer[0..nz]);
                }
                k = 0;
                while (k < nz) : (k += 1) {
                    data[x + nx * (y2 + ny * k)] = buffer[k];
                }
            }
        }
    }
};

// ============== Tests ==============

test "Plan1d power of two" {
    const allocator = std.testing.allocator;

    var plan = try Plan1d.init(allocator, 8);
    defer plan.deinit();

    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
        Complex.init(7, 0),
        Complex.init(8, 0),
    };

    const original = data;

    plan.forward(&data);
    plan.inverse(&data);

    for (data, original) |d, o| {
        try std.testing.expectApproxEqAbs(d.re, o.re, 1e-10);
        try std.testing.expectApproxEqAbs(d.im, o.im, 1e-10);
    }
}

test "Plan1d arbitrary size" {
    const allocator = std.testing.allocator;

    // Test size 6 (non-power of two)
    var plan = try Plan1d.init(allocator, 6);
    defer plan.deinit();

    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
    };

    const original = data;

    plan.forward(&data);
    plan.inverse(&data);

    for (data, original) |d, o| {
        try std.testing.expectApproxEqAbs(d.re, o.re, 1e-9);
        try std.testing.expectApproxEqAbs(d.im, o.im, 1e-9);
    }
}

test "Plan3d" {
    const allocator = std.testing.allocator;

    var plan = try Plan3d.init(allocator, 4, 4, 4);
    defer plan.deinit();

    var data: [64]Complex = undefined;
    var original: [64]Complex = undefined;
    for (0..64) |i| {
        data[i] = Complex.init(@floatFromInt(i), 0);
        original[i] = data[i];
    }

    plan.forward(&data);
    plan.inverse(&data);

    for (data, original) |d, o| {
        try std.testing.expectApproxEqAbs(d.re, o.re, 1e-10);
        try std.testing.expectApproxEqAbs(d.im, o.im, 1e-10);
    }
}
