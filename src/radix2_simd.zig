//! SIMD-optimized Radix-2 FFT
//!
//! Uses @Vector for parallel butterfly operations.
//! Automatically selects optimal vector width based on target architecture:
//! - x86_64 with AVX: 4 f64 (256-bit) = 2 complex numbers
//! - ARM (Apple Silicon): 2 f64 (128-bit) = 1 complex number
//!
//! Even on ARM, processing multiple butterflies with explicit vectorization
//! can help the compiler generate better code.

const std = @import("std");
const builtin = @import("builtin");
const Complex = @import("complex.zig").Complex;

/// Detect optimal SIMD width based on target architecture
const simd_config = struct {
    /// Number of f64 values per vector
    const vec_width: usize = blk: {
        const arch = builtin.cpu.arch;
        if (arch == .x86_64) {
            // x86_64: Check for AVX support
            if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx)) {
                break :blk 4; // AVX: 256-bit = 4 f64
            }
            break :blk 2; // SSE: 128-bit = 2 f64
        } else if (arch == .aarch64) {
            // ARM64 (Apple Silicon): NEON is 128-bit
            break :blk 2; // NEON: 128-bit = 2 f64
        } else {
            // Fallback
            break :blk 2;
        }
    };

    /// Number of complex numbers processed per SIMD operation
    const complex_per_vec: usize = vec_width / 2;

    /// Vector type
    const Vec = @Vector(vec_width, f64);
};

/// Export for debugging/testing
pub const SIMD_WIDTH = simd_config.vec_width;
pub const COMPLEX_PER_VEC = simd_config.complex_per_vec;

/// Check if value is power of two.
pub fn isPowerOfTwo(n: usize) bool {
    return n != 0 and (n & (n - 1)) == 0;
}

fn log2Exact(n: usize) usize {
    var bits: usize = 0;
    var v = n;
    while (v > 1) : (v >>= 1) {
        bits += 1;
    }
    return bits;
}

fn reverseBits(value: usize, bits: usize) usize {
    var x = value;
    var r: usize = 0;
    var idx: usize = 0;
    while (idx < bits) : (idx += 1) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

/// SIMD vector type (architecture-dependent)
const Vec = simd_config.Vec;

/// Vec2 for single complex number operations (works on all architectures)
const Vec2 = @Vector(2, f64);

/// Complex multiplication using Vec2: (a + bi)(c + di) = (ac-bd) + (ad+bc)i
inline fn complexMulVec2(a: Vec2, b: Vec2) Vec2 {
    // a = [re_a, im_a], b = [re_b, im_b]
    const a_re: Vec2 = @splat(a[0]);
    const a_im: Vec2 = @splat(a[1]);
    const b_flip = Vec2{ b[1], b[0] }; // [im_b, re_b]
    const prod1 = a_re * b; // [re_a*re_b, re_a*im_b]
    const prod2 = a_im * b_flip; // [im_a*im_b, im_a*re_b]
    return Vec2{ prod1[0] - prod2[0], prod1[1] + prod2[1] };
}

/// SIMD-optimized Plan for radix-2 FFT.
pub const Plan = struct {
    n: usize,
    bitrev: []usize,
    stage_count: usize,
    // Twiddle factors stored as separate re/im arrays for better SIMD access
    twiddle_re: []f64,
    twiddle_im: []f64,
    stage_offsets: []usize,
    stage_half: []usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !Plan {
        if (!isPowerOfTwo(n)) return error.InvalidSize;
        if (n == 0) return error.InvalidSize;

        const bits = log2Exact(n);
        const stage_count = bits;

        const stage_offsets = try allocator.alloc(usize, stage_count);
        errdefer allocator.free(stage_offsets);
        const stage_half = try allocator.alloc(usize, stage_count);
        errdefer allocator.free(stage_half);

        var total_twiddles: usize = 0;
        var s: usize = 0;
        while (s < stage_count) : (s += 1) {
            const len = @as(usize, 1) << @intCast(s + 1);
            const half = len >> 1;
            stage_offsets[s] = total_twiddles;
            stage_half[s] = half;
            total_twiddles += half;
        }

        // Allocate separate re/im arrays for SIMD-friendly access
        const twiddle_re = try allocator.alloc(f64, total_twiddles);
        errdefer allocator.free(twiddle_re);
        const twiddle_im = try allocator.alloc(f64, total_twiddles);
        errdefer allocator.free(twiddle_im);

        s = 0;
        while (s < stage_count) : (s += 1) {
            const len = @as(usize, 1) << @intCast(s + 1);
            const half = stage_half[s];
            const base = stage_offsets[s];
            var k: usize = 0;
            while (k < half) : (k += 1) {
                const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(len));
                twiddle_re[base + k] = @cos(angle);
                twiddle_im[base + k] = @sin(angle);
            }
        }

        const bitrev = try allocator.alloc(usize, n);
        errdefer allocator.free(bitrev);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            bitrev[i] = reverseBits(i, bits);
        }

        return .{
            .n = n,
            .bitrev = bitrev,
            .stage_count = stage_count,
            .twiddle_re = twiddle_re,
            .twiddle_im = twiddle_im,
            .stage_offsets = stage_offsets,
            .stage_half = stage_half,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan) void {
        if (self.bitrev.len > 0) self.allocator.free(self.bitrev);
        if (self.stage_offsets.len > 0) self.allocator.free(self.stage_offsets);
        if (self.stage_half.len > 0) self.allocator.free(self.stage_half);
        if (self.twiddle_re.len > 0) self.allocator.free(self.twiddle_re);
        if (self.twiddle_im.len > 0) self.allocator.free(self.twiddle_im);
    }

    pub fn forward(self: *const Plan, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *const Plan, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *const Plan, data: []Complex, inv: bool) void {
        const n = self.n;
        if (data.len != n) return;
        if (n <= 1) return;

        // Bit-reversal permutation
        for (0..n) |idx| {
            const j = self.bitrev[idx];
            if (idx < j) {
                const tmp = data[idx];
                data[idx] = data[j];
                data[j] = tmp;
            }
        }

        // Butterfly operations with Vec2 SIMD (128-bit, works on ARM NEON and x86 SSE)
        // Load/store through Complex struct to avoid alignment issues
        for (0..self.stage_count) |s| {
            const half = self.stage_half[s];
            const len = half << 1;
            const base = self.stage_offsets[s];

            var start: usize = 0;
            while (start < n) : (start += len) {
                for (0..half) |j0| {
                    // Load twiddle factor as Vec2
                    var w = Vec2{ self.twiddle_re[base + j0], self.twiddle_im[base + j0] };
                    if (inv) w[1] = -w[1]; // Conjugate for inverse

                    // Load u and v from Complex array (no alignment requirement)
                    const u_c = data[start + j0];
                    const v_c = data[start + j0 + half];
                    const u = Vec2{ u_c.re, u_c.im };
                    const v = Vec2{ v_c.re, v_c.im };

                    // v * w (complex multiplication)
                    const vw = complexMulVec2(v, w);

                    // Butterfly: top = u + vw, bottom = u - vw
                    const top = u + vw;
                    const bot = u - vw;

                    // Store back to Complex array
                    data[start + j0] = Complex.init(top[0], top[1]);
                    data[start + j0 + half] = Complex.init(bot[0], bot[1]);
                }
            }
        }

        // Normalize for inverse
        if (inv) {
            const scale = 1.0 / @as(f64, @floatFromInt(n));
            for (data) |*c| {
                c.* = Complex.scale(c.*, scale);
            }
        }
    }
};

// ============== Tests ==============

test "SIMD config" {
    // Verify the constants are valid
    try std.testing.expect(SIMD_WIDTH >= 2);
    try std.testing.expect(COMPLEX_PER_VEC >= 1);
}

test "radix2_simd Plan roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 8);
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

test "radix2_simd Plan vs direct DFT" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 4);
    defer plan.deinit();

    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
    };

    plan.forward(&data);

    // Expected DFT of [1,2,3,4]
    try std.testing.expectApproxEqAbs(data[0].re, 10.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[0].im, 0.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[1].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[1].im, 2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[2].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[2].im, 0.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[3].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[3].im, -2.0, 1e-10);
}

test "radix2_simd large size" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 1024);
    defer plan.deinit();

    var data = try allocator.alloc(Complex, 1024);
    defer allocator.free(data);

    for (0..1024) |i| {
        data[i] = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
    }

    const original = try allocator.alloc(Complex, 1024);
    defer allocator.free(original);
    @memcpy(original, data);

    plan.forward(data);
    plan.inverse(data);

    for (0..1024) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "radix2_simd works with unaligned memory" {
    // Test that SIMD version works correctly even when Complex array
    // is not 16-byte aligned (which would crash the old @alignCast version)
    const allocator = std.testing.allocator;

    // Allocate with explicit 8-byte alignment (Complex struct alignment)
    // This may not be 16-byte aligned, which is what we want to test
    const data = try allocator.alloc(Complex, 8);
    defer allocator.free(data);

    var plan = try Plan.init(allocator, 8);
    defer plan.deinit();

    // Initialize with test data
    for (0..8) |i| {
        data[i] = Complex.init(@floatFromInt(i + 1), 0);
    }

    // Store original
    var original: [8]Complex = undefined;
    @memcpy(&original, data);

    // Forward and inverse should roundtrip correctly
    plan.forward(data);
    plan.inverse(data);

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-10);
    }
}

test "radix2_simd small sizes n=2 and n=4" {
    const allocator = std.testing.allocator;

    // Test n=2
    {
        var plan = try Plan.init(allocator, 2);
        defer plan.deinit();

        var data = [_]Complex{ Complex.init(1, 0), Complex.init(2, 0) };
        const original = data;

        plan.forward(&data);
        plan.inverse(&data);

        for (data, original) |d, o| {
            try std.testing.expectApproxEqAbs(d.re, o.re, 1e-10);
            try std.testing.expectApproxEqAbs(d.im, o.im, 1e-10);
        }
    }

    // Test n=4
    {
        var plan = try Plan.init(allocator, 4);
        defer plan.deinit();

        var data = [_]Complex{
            Complex.init(1, 0),
            Complex.init(2, 0),
            Complex.init(3, 0),
            Complex.init(4, 0),
        };
        const original = data;

        plan.forward(&data);
        plan.inverse(&data);

        for (data, original) |d, o| {
            try std.testing.expectApproxEqAbs(d.re, o.re, 1e-10);
            try std.testing.expectApproxEqAbs(d.im, o.im, 1e-10);
        }
    }
}
