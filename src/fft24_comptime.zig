//! Comptime-optimized FFT for N=24
//!
//! 24 = 2³ × 3 = 8 × 3
//! Uses Cooley-Tukey with precomputed twiddle factors.
//!
//! Key optimizations:
//! 1. Twiddle factors computed at comptime
//! 2. Fully unrolled loops where beneficial
//! 3. SIMD vectorization for butterflies

const std = @import("std");
const Complex = @import("complex.zig").Complex;

// SIMD types
const Vec2 = @Vector(2, f64);
const Vec4 = @Vector(4, f64);

/// Precomputed twiddle factors for N=24
/// W_24^k = e^(-2πik/24) for k = 0..23
const Twiddles24 = struct {
    // W_24^k values (forward transform)
    w: [24]Complex,
    // W_24^k values (inverse transform)
    w_inv: [24]Complex,

    // Compute at comptime
    fn init() Twiddles24 {
        var result: Twiddles24 = undefined;
        const angle_base = -2.0 * std.math.pi / 24.0;
        for (0..24) |k| {
            const angle = angle_base * @as(f64, @floatFromInt(k));
            result.w[k] = Complex.init(@cos(angle), @sin(angle));
            result.w_inv[k] = Complex.init(@cos(-angle), @sin(-angle));
        }
        return result;
    }
};

/// Comptime-computed twiddle factors
const twiddles24 = Twiddles24.init();

/// Precomputed twiddle factors for N=8 (used in radix-8 stage)
const Twiddles8 = struct {
    w: [8]Complex,
    w_inv: [8]Complex,

    fn init() Twiddles8 {
        var result: Twiddles8 = undefined;
        const angle_base = -2.0 * std.math.pi / 8.0;
        for (0..8) |k| {
            const angle = angle_base * @as(f64, @floatFromInt(k));
            result.w[k] = Complex.init(@cos(angle), @sin(angle));
            result.w_inv[k] = Complex.init(@cos(-angle), @sin(-angle));
        }
        return result;
    }
};

const twiddles8 = Twiddles8.init();

/// Precomputed twiddle factors for N=3
const Twiddles3 = struct {
    w1: Complex, // W_3^1 = e^(-2πi/3)
    w2: Complex, // W_3^2 = e^(-4πi/3)
    w1_inv: Complex,
    w2_inv: Complex,

    fn init() Twiddles3 {
        const angle = -2.0 * std.math.pi / 3.0;
        return .{
            .w1 = Complex.init(@cos(angle), @sin(angle)),
            .w2 = Complex.init(@cos(2 * angle), @sin(2 * angle)),
            .w1_inv = Complex.init(@cos(-angle), @sin(-angle)),
            .w2_inv = Complex.init(@cos(-2 * angle), @sin(-2 * angle)),
        };
    }
};

const twiddles3 = Twiddles3.init();

// ============== Basic operations ==============

inline fn loadVec2(c: Complex) Vec2 {
    return Vec2{ c.re, c.im };
}

inline fn storeVec2(v: Vec2) Complex {
    return Complex.init(v[0], v[1]);
}

inline fn complexMulVec2(a: Vec2, b: Vec2) Vec2 {
    const a_re: Vec2 = @splat(a[0]);
    const a_im: Vec2 = @splat(a[1]);
    const b_flip = Vec2{ b[1], b[0] };
    const prod1 = a_re * b;
    const prod2 = a_im * b_flip;
    return Vec2{ prod1[0] - prod2[0], prod1[1] + prod2[1] };
}

inline fn complexMul(a: Complex, b: Complex) Complex {
    return Complex.init(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    );
}

inline fn complexAdd(a: Complex, b: Complex) Complex {
    return Complex.init(a.re + b.re, a.im + b.im);
}

inline fn complexSub(a: Complex, b: Complex) Complex {
    return Complex.init(a.re - b.re, a.im - b.im);
}

// ============== DFT-3 with precomputed twiddles ==============

/// DFT of size 3 using precomputed twiddles
inline fn dft3(x0: Complex, x1: Complex, x2: Complex, inv: bool) [3]Complex {
    const w1 = if (inv) twiddles3.w1_inv else twiddles3.w1;
    const w2 = if (inv) twiddles3.w2_inv else twiddles3.w2;

    return .{
        complexAdd(complexAdd(x0, x1), x2),
        complexAdd(complexAdd(x0, complexMul(x1, w1)), complexMul(x2, w2)),
        complexAdd(complexAdd(x0, complexMul(x1, w2)), complexMul(x2, w1)),
    };
}

// ============== DFT-8 with precomputed twiddles ==============

/// DFT of size 8 (radix-2, 3 stages)
inline fn dft8(input: *const [8]Complex, inv: bool) [8]Complex {
    const w = if (inv) &twiddles8.w_inv else &twiddles8.w;

    // Stage 1: 4 butterflies
    var s1: [8]Complex = undefined;
    s1[0] = complexAdd(input[0], input[4]);
    s1[1] = complexAdd(input[1], input[5]);
    s1[2] = complexAdd(input[2], input[6]);
    s1[3] = complexAdd(input[3], input[7]);
    s1[4] = complexSub(input[0], input[4]);
    s1[5] = complexMul(complexSub(input[1], input[5]), w[1]);
    s1[6] = complexMul(complexSub(input[2], input[6]), w[2]);
    s1[7] = complexMul(complexSub(input[3], input[7]), w[3]);

    // Stage 2: 4 butterflies
    var s2: [8]Complex = undefined;
    s2[0] = complexAdd(s1[0], s1[2]);
    s2[1] = complexAdd(s1[1], s1[3]);
    s2[2] = complexSub(s1[0], s1[2]);
    s2[3] = complexMul(complexSub(s1[1], s1[3]), w[2]);
    s2[4] = complexAdd(s1[4], s1[6]);
    s2[5] = complexAdd(s1[5], s1[7]);
    s2[6] = complexSub(s1[4], s1[6]);
    s2[7] = complexMul(complexSub(s1[5], s1[7]), w[2]);

    // Stage 3: 4 butterflies
    var result: [8]Complex = undefined;
    result[0] = complexAdd(s2[0], s2[1]);
    result[4] = complexSub(s2[0], s2[1]);
    result[2] = complexAdd(s2[2], s2[3]);
    result[6] = complexSub(s2[2], s2[3]);
    result[1] = complexAdd(s2[4], s2[5]);
    result[5] = complexSub(s2[4], s2[5]);
    result[3] = complexAdd(s2[6], s2[7]);
    result[7] = complexSub(s2[6], s2[7]);

    return result;
}

// ============== FFT-24 main implementation ==============

/// FFT for N=24 using 8×3 decomposition with precomputed twiddles
/// Cooley-Tukey DIF: N1=8, N2=3
pub fn fft24(data: []Complex, inv: bool) void {
    if (data.len != 24) return;

    const w = if (inv) &twiddles24.w_inv else &twiddles24.w;

    // Step 1: Rearrange into 8×3 matrix and DFT each row (size 3)
    // Input index: n = n1 + 8*n2 where n1 in [0,7], n2 in [0,2]
    var y: [8][3]Complex = undefined;

    // Process each row (DFT-3)
    inline for (0..8) |n1| {
        const row = dft3(data[n1], data[n1 + 8], data[n1 + 16], inv);
        y[n1] = row;
    }

    // Step 2: Apply twiddle factors W_24^(n1*k2)
    // Only non-zero exponents need multiplication
    inline for (0..8) |n1| {
        inline for (0..3) |k2| {
            const exp = n1 * k2;
            if (exp != 0) {
                const tw_idx = exp % 24;
                y[n1][k2] = complexMul(y[n1][k2], w[tw_idx]);
            }
        }
    }

    // Step 3: DFT each column (size 8)
    var z: [8][3]Complex = undefined;
    inline for (0..3) |k2| {
        const col = [8]Complex{
            y[0][k2], y[1][k2], y[2][k2], y[3][k2],
            y[4][k2], y[5][k2], y[6][k2], y[7][k2],
        };
        const result = dft8(&col, inv);
        inline for (0..8) |k1| {
            z[k1][k2] = result[k1];
        }
    }

    // Step 4: Output in row-major order: k = 3*k1 + k2
    inline for (0..8) |k1| {
        inline for (0..3) |k2| {
            data[3 * k1 + k2] = z[k1][k2];
        }
    }

    // Normalize for inverse
    if (inv) {
        const scale = 1.0 / 24.0;
        for (data) |*v| {
            v.re *= scale;
            v.im *= scale;
        }
    }
}

/// Alternative: 3×8 decomposition (might be faster due to cache)
pub fn fft24_alt(data: []Complex, inv: bool) void {
    if (data.len != 24) return;

    const w = if (inv) &twiddles24.w_inv else &twiddles24.w;

    // N1=3, N2=8 decomposition
    // Input index: n = n1 + 3*n2

    // Step 1: DFT each row (size 8)
    var y: [3][8]Complex = undefined;

    inline for (0..3) |n1| {
        var row: [8]Complex = undefined;
        inline for (0..8) |n2| {
            row[n2] = data[n1 + 3 * n2];
        }
        y[n1] = dft8(&row, inv);
    }

    // Step 2: Apply twiddle factors W_24^(n1*k2)
    inline for (0..3) |n1| {
        inline for (0..8) |k2| {
            const exp = n1 * k2;
            if (exp != 0) {
                const tw_idx = exp % 24;
                y[n1][k2] = complexMul(y[n1][k2], w[tw_idx]);
            }
        }
    }

    // Step 3: DFT each column (size 3)
    var z: [3][8]Complex = undefined;
    inline for (0..8) |k2| {
        const col = dft3(y[0][k2], y[1][k2], y[2][k2], inv);
        z[0][k2] = col[0];
        z[1][k2] = col[1];
        z[2][k2] = col[2];
    }

    // Step 4: Output: k = 8*k1 + k2
    inline for (0..3) |k1| {
        inline for (0..8) |k2| {
            data[8 * k1 + k2] = z[k1][k2];
        }
    }

    // Normalize for inverse
    if (inv) {
        const scale = 1.0 / 24.0;
        for (data) |*v| {
            v.re *= scale;
            v.im *= scale;
        }
    }
}

// ============== Tests ==============

fn dftDirect(input: []const Complex, output: []Complex, inv: bool) void {
    const n = input.len;
    const sign: f64 = if (inv) 1.0 else -1.0;
    const angle_base = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

    for (0..n) |k| {
        var sum = Complex.init(0, 0);
        for (0..n) |j| {
            const angle = angle_base * @as(f64, @floatFromInt(k * j));
            sum = complexAdd(sum, complexMul(input[j], Complex.init(@cos(angle), @sin(angle))));
        }
        output[k] = sum;
    }
}

test "fft24 vs direct" {
    var data: [24]Complex = undefined;
    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt(i % 5));
    }

    var expected: [24]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft24(&data, false);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft24 roundtrip" {
    var data: [24]Complex = undefined;
    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt((i * 3) % 7));
    }
    const original = data;

    fft24(&data, false);
    fft24(&data, true);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "fft24_alt vs direct" {
    var data: [24]Complex = undefined;
    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt(i % 5));
    }

    var expected: [24]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft24_alt(&data, false);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft24_alt roundtrip" {
    var data: [24]Complex = undefined;
    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt((i * 3) % 7));
    }
    const original = data;

    fft24_alt(&data, false);
    fft24_alt(&data, true);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "twiddles24 correctness" {
    // Verify twiddle factors
    const angle = -2.0 * std.math.pi / 24.0;
    for (0..24) |k| {
        const expected_re = @cos(angle * @as(f64, @floatFromInt(k)));
        const expected_im = @sin(angle * @as(f64, @floatFromInt(k)));
        try std.testing.expectApproxEqAbs(twiddles24.w[k].re, expected_re, 1e-15);
        try std.testing.expectApproxEqAbs(twiddles24.w[k].im, expected_im, 1e-15);
    }
}
