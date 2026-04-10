//! vDSP FFT wrapper for Apple Accelerate framework
//!
//! This module provides FFT using Apple's vDSP library which is highly optimized
//! for Apple Silicon (M1/M2/M3) and Intel chips on macOS.
//!
//! vDSP uses split-complex format (separate real/imag arrays) internally,
//! so we need to convert from/to interleaved format.

const std = @import("std");
pub const Complex = @import("complex.zig").Complex;

// vDSP type definitions
const FFTSetupD = ?*anyopaque;
const vDSP_Length = c_ulong;
const vDSP_Stride = c_long;
const FFTDirection = c_int;

const FFT_FORWARD: FFTDirection = 1;
const FFT_INVERSE: FFTDirection = -1;

// Split complex format used by vDSP
const DSPDoubleSplitComplex = extern struct {
    realp: [*]f64,
    imagp: [*]f64,
};

// Radix hint for FFT setup
const FFTRadix = c_int;
const kFFTRadix2: FFTRadix = 0;
const kFFTRadix3: FFTRadix = 1;
const kFFTRadix5: FFTRadix = 2;

// vDSP function declarations
extern fn vDSP_create_fftsetupD(log2n: vDSP_Length, radix: FFTRadix) FFTSetupD;
extern fn vDSP_destroy_fftsetupD(setup: FFTSetupD) void;
extern fn vDSP_fft_zipD(
    setup: FFTSetupD,
    c: *DSPDoubleSplitComplex,
    stride: vDSP_Stride,
    log2n: vDSP_Length,
    direction: FFTDirection,
) void;

// For converting between interleaved and split complex
extern fn vDSP_ctozD(
    c: [*]const f64, // interleaved complex as f64 pairs
    ic: vDSP_Stride,
    z: *DSPDoubleSplitComplex,
    iz: vDSP_Stride,
    n: vDSP_Length,
) void;

extern fn vDSP_ztocD(
    z: *const DSPDoubleSplitComplex,
    iz: vDSP_Stride,
    c: [*]f64,
    ic: vDSP_Stride,
    n: vDSP_Length,
) void;

/// Check if n is a power of two
fn isPowerOfTwo(n: usize) bool {
    return n > 0 and (n & (n - 1)) == 0;
}

/// Calculate log2 of a power of two
fn log2(n: usize) usize {
    if (n == 0) return 0;
    var val = n;
    var result: usize = 0;
    while (val > 1) {
        val >>= 1;
        result += 1;
    }
    return result;
}

/// 1D FFT Plan using vDSP
pub const VdspPlan1d = struct {
    n: usize,
    log2n: usize,
    setup: FFTSetupD,
    split_real: []f64,
    split_imag: []f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !VdspPlan1d {
        if (n == 0) return error.InvalidSize;
        if (!isPowerOfTwo(n)) return error.NotPowerOfTwo;

        const log2n = log2(n);
        const setup = vDSP_create_fftsetupD(@intCast(log2n), kFFTRadix2);
        if (setup == null) return error.VdspSetupFailed;

        const split_real = try allocator.alloc(f64, n);
        errdefer allocator.free(split_real);

        const split_imag = try allocator.alloc(f64, n);
        errdefer allocator.free(split_imag);

        return .{
            .n = n,
            .log2n = log2n,
            .setup = setup,
            .split_real = split_real,
            .split_imag = split_imag,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VdspPlan1d) void {
        if (self.setup != null) {
            vDSP_destroy_fftsetupD(self.setup);
        }
        self.allocator.free(self.split_real);
        self.allocator.free(self.split_imag);
    }

    pub fn forward(self: *VdspPlan1d, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *VdspPlan1d, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *VdspPlan1d, data: []Complex, inv: bool) void {
        if (data.len != self.n) return;

        // Convert interleaved to split complex
        var split = DSPDoubleSplitComplex{
            .realp = self.split_real.ptr,
            .imagp = self.split_imag.ptr,
        };

        // vDSP_ctozD expects interleaved data as f64 pairs with stride 2
        const data_ptr: [*]const f64 = @ptrCast(data.ptr);
        vDSP_ctozD(data_ptr, 2, &split, 1, @intCast(self.n));

        // Perform FFT
        const direction: FFTDirection = if (inv) FFT_INVERSE else FFT_FORWARD;
        vDSP_fft_zipD(self.setup, &split, 1, @intCast(self.log2n), direction);

        // Convert split to interleaved
        const out_ptr: [*]f64 = @ptrCast(data.ptr);
        vDSP_ztocD(&split, 1, out_ptr, 2, @intCast(self.n));

        // vDSP does not normalize inverse FFT, we need to scale
        if (inv) {
            const scale = 1.0 / @as(f64, @floatFromInt(self.n));
            for (data) |*v| {
                v.* = Complex.scale(v.*, scale);
            }
        }
    }
};

/// 3D FFT Plan using vDSP
pub const VdspPlan3d = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    plan_x: VdspPlan1d,
    plan_y: VdspPlan1d,
    plan_z: VdspPlan1d,
    scratch: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !VdspPlan3d {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        var plan_x = try VdspPlan1d.init(allocator, nx);
        errdefer plan_x.deinit();

        var plan_y = try VdspPlan1d.init(allocator, ny);
        errdefer plan_y.deinit();

        var plan_z = try VdspPlan1d.init(allocator, nz);
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

    pub fn deinit(self: *VdspPlan3d) void {
        self.plan_x.deinit();
        self.plan_y.deinit();
        self.plan_z.deinit();
        if (self.scratch.len > 0) self.allocator.free(self.scratch);
    }

    pub fn forward(self: *VdspPlan3d, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *VdspPlan3d, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *VdspPlan3d, data: []Complex, inv: bool) void {
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

        // FFT along y-axis (requires data gather/scatter)
        const buffer = self.scratch;
        for (0..nx) |x| {
            for (0..nz) |z| {
                // Gather y-axis data
                for (0..ny) |j| {
                    buffer[j] = data[x + nx * (j + ny * z)];
                }
                if (inv) {
                    self.plan_y.inverse(buffer[0..ny]);
                } else {
                    self.plan_y.forward(buffer[0..ny]);
                }
                // Scatter back
                for (0..ny) |j| {
                    data[x + nx * (j + ny * z)] = buffer[j];
                }
            }
        }

        // FFT along z-axis (requires data gather/scatter)
        for (0..nx) |x| {
            for (0..ny) |y| {
                // Gather z-axis data
                for (0..nz) |k| {
                    buffer[k] = data[x + nx * (y + ny * k)];
                }
                if (inv) {
                    self.plan_z.inverse(buffer[0..nz]);
                } else {
                    self.plan_z.forward(buffer[0..nz]);
                }
                // Scatter back
                for (0..nz) |k| {
                    data[x + nx * (y + ny * k)] = buffer[k];
                }
            }
        }
    }
};

// ============== Tests ==============

test "VdspPlan1d roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try VdspPlan1d.init(allocator, 8);
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

test "VdspPlan3d roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try VdspPlan3d.init(allocator, 4, 4, 4);
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
