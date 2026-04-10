//! Mixed-Radix FFT Algorithm
//!
//! Step-by-step implementation with careful testing.
//! Starting with N=6 (2×3) as the simplest composite case.

const std = @import("std");
const Complex = @import("complex.zig").Complex;
const bluestein = @import("bluestein.zig");

// SIMD types for ARM NEON / x86 SSE (128-bit = 2 x f64)
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

/// Load Complex as Vec2
inline fn loadVec2(c: Complex) Vec2 {
    return Vec2{ c.re, c.im };
}

/// Store Vec2 as Complex
inline fn storeVec2(v: Vec2) Complex {
    return Complex.init(v[0], v[1]);
}

/// Check if n is a smooth number (composed only of factors 2, 3, 5).
pub fn isSmoothNumber(n: usize) bool {
    if (n == 0) return false;
    var v = n;
    for ([_]usize{ 2, 3, 5 }) |f| {
        while (v % f == 0) {
            v /= f;
        }
    }
    return v == 1;
}

/// Direct DFT - O(N^2) reference implementation for testing.
pub fn dftDirect(input: []const Complex, output: []Complex, inv: bool) void {
    const n = input.len;
    const sign: f64 = if (inv) 1.0 else -1.0;
    const angle_base = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

    for (0..n) |k| {
        var sum = Complex.zero;
        for (0..n) |j| {
            const angle = angle_base * @as(f64, @floatFromInt(k * j));
            sum = Complex.add(sum, Complex.mul(input[j], Complex.expi(angle)));
        }
        output[k] = sum;
    }
}

/// FFT for N=6 using Cooley-Tukey decomposition.
/// N = 2 × 3: First do 2 DFTs of size 3, then twiddle, then 3 DFTs of size 2.
///
/// Algorithm:
///   Input x[n] where n = 0..5
///   View as 2×3 matrix (2 rows, 3 cols):
///     x[0] x[2] x[4]   (row 0: even indices)
///     x[1] x[3] x[5]   (row 1: odd indices)
///
///   Step 1: DFT each row (2 DFTs of size 3)
///   Step 2: Multiply by twiddle factors W_6^(row * col)
///   Step 3: DFT each column (3 DFTs of size 2)
///   Step 4: Read out in column-major order
pub fn fft6(data: []Complex, inv: bool) void {
    if (data.len != 6) return;

    // Step 1: Rearrange into 2×3 matrix and DFT rows
    // Row 0: x[0], x[2], x[4] → DFT → y[0,0], y[0,1], y[0,2]
    // Row 1: x[1], x[3], x[5] → DFT → y[1,0], y[1,1], y[1,2]

    var y: [2][3]Complex = undefined;

    // DFT of size 3 for row 0
    {
        const row = [3]Complex{ data[0], data[2], data[4] };
        var out: [3]Complex = undefined;
        dft3(&row, &out, inv);
        y[0] = out;
    }

    // DFT of size 3 for row 1
    {
        const row = [3]Complex{ data[1], data[3], data[5] };
        var out: [3]Complex = undefined;
        dft3(&row, &out, inv);
        y[1] = out;
    }

    // Step 2: Apply twiddle factors W_6^(row * col)
    // W_6 = e^(-2πi/6) = e^(-πi/3)
    // Only row 1 needs twiddles (row 0 has W^0 = 1)
    const sign: f64 = if (inv) 1.0 else -1.0;
    const w6_angle = sign * std.math.pi / 3.0; // -2π/6 = -π/3

    for (0..3) |col| {
        // y[1][col] *= W_6^(1 * col) = W_6^col
        const tw = Complex.expi(w6_angle * @as(f64, @floatFromInt(col)));
        y[1][col] = Complex.mul(y[1][col], tw);
    }

    // Step 3: DFT each column (3 DFTs of size 2)
    var z: [2][3]Complex = undefined;
    for (0..3) |col| {
        const column = [2]Complex{ y[0][col], y[1][col] };
        var out: [2]Complex = undefined;
        dft2(&column, &out, inv);
        z[0][col] = out[0];
        z[1][col] = out[1];
    }

    // Step 4: Output in row-major order (DIF convention)
    // k = N2*k1 + k2 = 3*k1 + k2, where k1 in {0,1}, k2 in {0,1,2}
    // X[0] = z[0][0], X[1] = z[0][1], X[2] = z[0][2]
    // X[3] = z[1][0], X[4] = z[1][1], X[5] = z[1][2]
    for (0..2) |row| {
        for (0..3) |col| {
            data[row * 3 + col] = z[row][col];
        }
    }

    // Normalize for inverse
    if (inv) {
        for (data) |*v| {
            v.* = Complex.scale(v.*, 1.0 / 6.0);
        }
    }
}

/// DFT of size 2 (butterfly).
fn dft2(input: *const [2]Complex, output: *[2]Complex, inv: bool) void {
    _ = inv; // Same for forward and inverse
    output[0] = Complex.add(input[0], input[1]);
    output[1] = Complex.sub(input[0], input[1]);
}

/// DFT of size 3 (in-place version for general FFT).
fn dft3InPlace(data: []Complex, inv: bool) void {
    if (data.len != 3) return;
    const input = [3]Complex{ data[0], data[1], data[2] };
    var output: [3]Complex = undefined;
    dft3(&input, &output, inv);
    data[0] = output[0];
    data[1] = output[1];
    data[2] = output[2];
}

/// DFT of size 3 - SIMD optimized.
fn dft3(input: *const [3]Complex, output: *[3]Complex, inv: bool) void {
    // W3 = e^(-2πi/3) = -1/2 - i*sqrt(3)/2
    const sign: f64 = if (inv) 1.0 else -1.0;
    const sqrt3_2: f64 = 0.8660254037844386;

    const w3 = Vec2{ -0.5, sign * sqrt3_2 };
    const w3_2 = Vec2{ -0.5, -sign * sqrt3_2 }; // W3^2 = conj(W3)

    const x0 = loadVec2(input[0]);
    const x1 = loadVec2(input[1]);
    const x2 = loadVec2(input[2]);

    // X[0] = x0 + x1 + x2
    // X[1] = x0 + x1*W3 + x2*W3^2
    // X[2] = x0 + x1*W3^2 + x2*W3
    output[0] = storeVec2(x0 + x1 + x2);
    output[1] = storeVec2(x0 + complexMulVec2(x1, w3) + complexMulVec2(x2, w3_2));
    output[2] = storeVec2(x0 + complexMulVec2(x1, w3_2) + complexMulVec2(x2, w3));
}

/// DFT of size 5 - SIMD optimized.
fn dft5(input: *const [5]Complex, output: *[5]Complex, inv: bool) void {
    // W5 = e^(-2πi/5)
    // cos(2π/5) = (√5-1)/4 ≈ 0.309016994
    // sin(2π/5) = √(10+2√5)/4 ≈ 0.951056516
    // cos(4π/5) = -(√5+1)/4 ≈ -0.809016994
    // sin(4π/5) = √(10-2√5)/4 ≈ 0.587785252
    const sign: f64 = if (inv) 1.0 else -1.0;

    const c1: f64 = 0.30901699437494742; // cos(2π/5)
    const s1: f64 = 0.95105651629515353; // sin(2π/5)
    const c2: f64 = -0.80901699437494742; // cos(4π/5)
    const s2: f64 = 0.58778525229247313; // sin(4π/5)

    const w5_1 = Vec2{ c1, sign * s1 }; // W5^1
    const w5_2 = Vec2{ c2, sign * s2 }; // W5^2
    const w5_3 = Vec2{ c2, -sign * s2 }; // W5^3 = conj(W5^2)
    const w5_4 = Vec2{ c1, -sign * s1 }; // W5^4 = conj(W5^1)

    const x0 = loadVec2(input[0]);
    const x1 = loadVec2(input[1]);
    const x2 = loadVec2(input[2]);
    const x3 = loadVec2(input[3]);
    const x4 = loadVec2(input[4]);

    // X[k] = sum_{n=0}^{4} x[n] * W5^(nk)
    output[0] = storeVec2(x0 + x1 + x2 + x3 + x4);
    output[1] = storeVec2(x0 + complexMulVec2(x1, w5_1) + complexMulVec2(x2, w5_2) + complexMulVec2(x3, w5_3) + complexMulVec2(x4, w5_4));
    output[2] = storeVec2(x0 + complexMulVec2(x1, w5_2) + complexMulVec2(x2, w5_4) + complexMulVec2(x3, w5_1) + complexMulVec2(x4, w5_3));
    output[3] = storeVec2(x0 + complexMulVec2(x1, w5_3) + complexMulVec2(x2, w5_1) + complexMulVec2(x3, w5_4) + complexMulVec2(x4, w5_2));
    output[4] = storeVec2(x0 + complexMulVec2(x1, w5_4) + complexMulVec2(x2, w5_3) + complexMulVec2(x3, w5_2) + complexMulVec2(x4, w5_1));
}

/// DFT of size 5 (in-place version).
fn dft5InPlace(data: []Complex, inv: bool) void {
    if (data.len != 5) return;
    const input = [5]Complex{ data[0], data[1], data[2], data[3], data[4] };
    var output: [5]Complex = undefined;
    dft5(&input, &output, inv);
    for (0..5) |i| {
        data[i] = output[i];
    }
}

/// DFT of size 2 (in-place version) - SIMD optimized.
fn dft2InPlace(data: []Complex, inv: bool) void {
    _ = inv;
    if (data.len != 2) return;
    const a = loadVec2(data[0]);
    const b = loadVec2(data[1]);
    data[0] = storeVec2(a + b);
    data[1] = storeVec2(a - b);
}

/// Small DFT dispatcher - calls the appropriate specialized DFT.
fn smallDft(data: []Complex, inv: bool) void {
    switch (data.len) {
        1 => {}, // Nothing to do
        2 => dft2InPlace(data, inv),
        3 => dft3InPlace(data, inv),
        5 => dft5InPlace(data, inv),
        else => {
            // Fallback to direct DFT for other small sizes
            var output: [64]Complex = undefined;
            dftDirect(data, output[0..data.len], inv);
            for (0..data.len) |i| {
                data[i] = output[i];
            }
        },
    }
}

/// FFT for N=10 (2×5) using Cooley-Tukey DIF.
pub fn fft10(data: []Complex, inv: bool) void {
    if (data.len != 10) return;
    const N1: usize = 2;
    const N2: usize = 5;

    // Input: n = n1 + N1*n2 = n1 + 2*n2
    // Matrix[n1][n2] = data[n1 + 2*n2]
    var y: [N1][N2]Complex = undefined;

    // Step 1: Arrange into matrix and DFT each row (size N2=5)
    for (0..N1) |n1| {
        for (0..N2) |n2| {
            y[n1][n2] = data[n1 + N1 * n2];
        }
        // DFT row n1
        var row: [N2]Complex = y[n1];
        dft5InPlace(&row, inv);
        y[n1] = row;
    }

    // Step 2: Apply twiddle factors W_10^(n1*k2)
    const sign: f64 = if (inv) 1.0 else -1.0;
    const angle_base = sign * 2.0 * std.math.pi / 10.0;

    for (0..N1) |n1| {
        for (0..N2) |k2| {
            if (n1 * k2 != 0) { // W^0 = 1
                const tw = Complex.expi(angle_base * @as(f64, @floatFromInt(n1 * k2)));
                y[n1][k2] = Complex.mul(y[n1][k2], tw);
            }
        }
    }

    // Step 3: DFT each column (size N1=2)
    var z: [N1][N2]Complex = undefined;
    for (0..N2) |k2| {
        var col = [N1]Complex{ y[0][k2], y[1][k2] };
        dft2InPlace(&col, inv);
        z[0][k2] = col[0];
        z[1][k2] = col[1];
    }

    // Step 4: Output in row-major order: k = N2*k1 + k2
    for (0..N1) |k1| {
        for (0..N2) |k2| {
            data[N2 * k1 + k2] = z[k1][k2];
        }
    }

    // Normalize for inverse
    if (inv) {
        for (data) |*v| {
            v.* = Complex.scale(v.*, 1.0 / 10.0);
        }
    }
}

/// FFT for N=9 (3×3) using Cooley-Tukey DIF.
pub fn fft9(data: []Complex, inv: bool) void {
    if (data.len != 9) return;
    const N1: usize = 3;
    const N2: usize = 3;

    // Input: n = n1 + N1*n2 = n1 + 3*n2
    var y: [N1][N2]Complex = undefined;

    // Step 1: Arrange into matrix and DFT each row (size N2=3)
    for (0..N1) |n1| {
        for (0..N2) |n2| {
            y[n1][n2] = data[n1 + N1 * n2];
        }
        // DFT row n1
        var row: [N2]Complex = y[n1];
        dft3InPlace(&row, inv);
        y[n1] = row;
    }

    // Step 2: Apply twiddle factors W_9^(n1*k2)
    const sign: f64 = if (inv) 1.0 else -1.0;
    const angle_base = sign * 2.0 * std.math.pi / 9.0;

    for (0..N1) |n1| {
        for (0..N2) |k2| {
            if (n1 * k2 != 0) {
                const tw = Complex.expi(angle_base * @as(f64, @floatFromInt(n1 * k2)));
                y[n1][k2] = Complex.mul(y[n1][k2], tw);
            }
        }
    }

    // Step 3: DFT each column (size N1=3)
    var z: [N1][N2]Complex = undefined;
    for (0..N2) |k2| {
        var col = [N1]Complex{ y[0][k2], y[1][k2], y[2][k2] };
        dft3InPlace(&col, inv);
        for (0..N1) |k1| {
            z[k1][k2] = col[k1];
        }
    }

    // Step 4: Output in row-major order: k = N2*k1 + k2
    for (0..N1) |k1| {
        for (0..N2) |k2| {
            data[N2 * k1 + k2] = z[k1][k2];
        }
    }

    // Normalize for inverse
    if (inv) {
        for (data) |*v| {
            v.* = Complex.scale(v.*, 1.0 / 9.0);
        }
    }
}

/// FFT for N=12 (4×3 = 2²×3) using Cooley-Tukey DIF.
/// Using N1=4, N2=3 decomposition.
pub fn fft12(data: []Complex, inv: bool) void {
    if (data.len != 12) return;
    const N1: usize = 4;
    const N2: usize = 3;

    // Input: n = n1 + N1*n2 = n1 + 4*n2
    var y: [N1][N2]Complex = undefined;

    // Step 1: Arrange into matrix and DFT each row (size N2=3)
    for (0..N1) |n1| {
        for (0..N2) |n2| {
            y[n1][n2] = data[n1 + N1 * n2];
        }
        // DFT row n1
        var row: [N2]Complex = y[n1];
        dft3InPlace(&row, inv);
        y[n1] = row;
    }

    // Step 2: Apply twiddle factors W_12^(n1*k2)
    const sign: f64 = if (inv) 1.0 else -1.0;
    const angle_base = sign * 2.0 * std.math.pi / 12.0;

    for (0..N1) |n1| {
        for (0..N2) |k2| {
            if (n1 * k2 != 0) {
                const tw = Complex.expi(angle_base * @as(f64, @floatFromInt(n1 * k2)));
                y[n1][k2] = Complex.mul(y[n1][k2], tw);
            }
        }
    }

    // Step 3: DFT each column (size N1=4, use radix-2 butterfly)
    var z: [N1][N2]Complex = undefined;
    for (0..N2) |k2| {
        var col = [N1]Complex{ y[0][k2], y[1][k2], y[2][k2], y[3][k2] };
        dft4InPlace(&col, inv);
        for (0..N1) |k1| {
            z[k1][k2] = col[k1];
        }
    }

    // Step 4: Output in row-major order: k = N2*k1 + k2
    for (0..N1) |k1| {
        for (0..N2) |k2| {
            data[N2 * k1 + k2] = z[k1][k2];
        }
    }

    // Normalize for inverse
    if (inv) {
        for (data) |*v| {
            v.* = Complex.scale(v.*, 1.0 / 12.0);
        }
    }
}

/// DFT of size 4 using radix-2 (two stages of butterflies) - SIMD optimized.
fn dft4InPlace(data: []Complex, inv: bool) void {
    if (data.len != 4) return;

    // Stage 1: 2 butterflies
    const a0 = loadVec2(data[0]);
    const a1 = loadVec2(data[1]);
    const a2 = loadVec2(data[2]);
    const a3 = loadVec2(data[3]);

    // First level: pairs (0,2) and (1,3)
    const b0 = a0 + a2;
    const b2 = a0 - a2;
    const b1 = a1 + a3;
    const b3 = a1 - a3;

    // Apply twiddle for b3: W_4^1 = -i (forward) or +i (inverse)
    // -i * (re + im*i) = im - re*i → [im, -re]
    // +i * (re + im*i) = -im + re*i → [-im, re]
    const sign: f64 = if (inv) 1.0 else -1.0;
    const b3_tw = Vec2{ -b3[1] * sign, b3[0] * sign };

    // Stage 2: final butterflies
    data[0] = storeVec2(b0 + b1); // X[0]
    data[1] = storeVec2(b2 + b3_tw); // X[1]
    data[2] = storeVec2(b0 - b1); // X[2]
    data[3] = storeVec2(b2 - b3_tw); // X[3]
}

/// General mixed-radix FFT for smooth numbers (2^a × 3^b × 5^c).
/// This is a recursive implementation that factorizes N and applies Cooley-Tukey.
pub fn mixedRadixFft(data: []Complex, scratch: []Complex, inv: bool) void {
    const n = data.len;
    if (n <= 1) return;

    // Use specialized implementations for small sizes
    switch (n) {
        2 => dft2InPlace(data, inv),
        3 => dft3InPlace(data, inv),
        4 => dft4InPlace(data, inv),
        5 => dft5InPlace(data, inv),
        6 => fft6(data, inv),
        9 => fft9(data, inv),
        10 => fft10(data, inv),
        12 => fft12(data, inv),
        else => {
            // Find a factor
            var factor: usize = 0;
            for ([_]usize{ 2, 3, 5 }) |f| {
                if (n % f == 0) {
                    factor = f;
                    break;
                }
            }

            if (factor == 0) {
                // Not a smooth number, fall back to direct DFT
                dftDirect(data, scratch[0..n], inv);
                for (0..n) |i| {
                    data[i] = scratch[i];
                }
                return;
            }

            // N = N1 × N2 where N1 = factor
            const N1 = factor;
            const N2 = n / factor;

            // DIF: n = n1 + N1*n2, k = N2*k1 + k2
            // Step 1: Copy to scratch in matrix form and DFT rows (size N2)
            for (0..N1) |n1| {
                // Copy row n1 to scratch
                for (0..N2) |n2| {
                    scratch[n2] = data[n1 + N1 * n2];
                }
                // Recursive DFT of size N2
                mixedRadixFft(scratch[0..N2], scratch[N2..], inv);
                // Copy back
                for (0..N2) |n2| {
                    data[n1 + N1 * n2] = scratch[n2];
                }
            }

            // Step 2: Apply twiddle factors W_N^(n1*k2)
            const sign: f64 = if (inv) 1.0 else -1.0;
            const angle_base = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

            for (0..N1) |n1| {
                for (0..N2) |k2| {
                    if (n1 * k2 != 0) {
                        const tw = Complex.expi(angle_base * @as(f64, @floatFromInt(n1 * k2)));
                        const idx = n1 + N1 * k2;
                        data[idx] = Complex.mul(data[idx], tw);
                    }
                }
            }

            // Step 3: DFT each column (size N1)
            for (0..N2) |k2| {
                // Copy column to scratch
                for (0..N1) |n1| {
                    scratch[n1] = data[n1 + N1 * k2];
                }
                // Recursive DFT of size N1
                mixedRadixFft(scratch[0..N1], scratch[N1..], inv);
                // Copy back to output positions: k = N2*k1 + k2
                for (0..N1) |k1| {
                    data[N2 * k1 + k2] = scratch[k1];
                }
            }

            // Normalize for inverse at top level only
            // Note: This is tricky for recursive calls. We'll handle normalization
            // at the top level in the public API.
        },
    }
}

/// Public API for mixed-radix FFT with automatic memory allocation.
pub fn mixedRadixFftAlloc(alloc: std.mem.Allocator, data: []Complex, inv: bool) !void {
    const n = data.len;
    if (!isSmoothNumber(n)) {
        return error.NotSmoothNumber;
    }

    // Allocate scratch space (need 2*n for recursive calls)
    const scratch = try alloc.alloc(Complex, 2 * n);
    defer alloc.free(scratch);

    // Perform FFT without normalization
    mixedRadixFftNoNorm(data, scratch, inv);

    // Apply normalization for inverse
    if (inv) {
        const scale = 1.0 / @as(f64, @floatFromInt(n));
        for (data) |*v| {
            v.* = Complex.scale(v.*, scale);
        }
    }
}

/// Mixed-radix FFT without normalization.
/// Used internally and by fft.zig Plan1d wrapper.
pub fn mixedRadixFftNoNorm(data: []Complex, scratch: []Complex, inv: bool) void {
    const n = data.len;
    if (n <= 1) return;

    // Use specialized implementations for small sizes (these do NOT normalize)
    switch (n) {
        2 => dft2InPlace(data, inv),
        3 => dft3InPlace(data, inv),
        4 => dft4InPlace(data, inv),
        5 => dft5InPlace(data, inv),
        else => {
            // Find a factor
            var factor: usize = 0;
            for ([_]usize{ 2, 3, 5 }) |f| {
                if (n % f == 0) {
                    factor = f;
                    break;
                }
            }

            if (factor == 0) {
                // Not a smooth number, fall back to direct DFT (no normalization)
                dftDirect(data, scratch[0..n], inv);
                for (0..n) |i| {
                    data[i] = scratch[i];
                }
                return;
            }

            const N1 = factor;
            const N2 = n / factor;

            // Step 1: DFT each row
            for (0..N1) |n1| {
                for (0..N2) |n2| {
                    scratch[n2] = data[n1 + N1 * n2];
                }
                mixedRadixFftNoNorm(scratch[0..N2], scratch[N2..], inv);
                for (0..N2) |n2| {
                    data[n1 + N1 * n2] = scratch[n2];
                }
            }

            // Step 2: Apply twiddle factors
            const sign: f64 = if (inv) 1.0 else -1.0;
            const angle_base = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

            for (0..N1) |n1| {
                for (0..N2) |k2| {
                    if (n1 * k2 != 0) {
                        const tw = Complex.expi(angle_base * @as(f64, @floatFromInt(n1 * k2)));
                        const idx = n1 + N1 * k2;
                        data[idx] = Complex.mul(data[idx], tw);
                    }
                }
            }

            // Step 3: DFT each column
            // Use scratch[n..2n] as temporary output to avoid overwriting input
            const output = scratch[n .. 2 * n];

            for (0..N2) |k2| {
                for (0..N1) |n1| {
                    scratch[n1] = data[n1 + N1 * k2];
                }
                mixedRadixFftNoNorm(scratch[0..N1], scratch[N1..n], inv);
                // Store to temporary output at correct positions
                for (0..N1) |k1| {
                    output[N2 * k1 + k2] = scratch[k1];
                }
            }

            // Copy output back to data
            for (0..n) |i| {
                data[i] = output[i];
            }
        },
    }
}

// ============== Scalar versions for benchmarking ==============

/// DFT of size 2 (scalar version).
fn dft2InPlaceScalar(data: []Complex, inv: bool) void {
    _ = inv;
    if (data.len != 2) return;
    const a = data[0];
    const b = data[1];
    data[0] = Complex.add(a, b);
    data[1] = Complex.sub(a, b);
}

/// DFT of size 3 (scalar version).
fn dft3InPlaceScalar(data: []Complex, inv: bool) void {
    if (data.len != 3) return;
    const sign: f64 = if (inv) 1.0 else -1.0;
    const sqrt3_2: f64 = 0.8660254037844386;

    const w3 = Complex.init(-0.5, sign * sqrt3_2);
    const w3_2 = Complex.init(-0.5, -sign * sqrt3_2);

    const x0 = data[0];
    const x1 = data[1];
    const x2 = data[2];

    data[0] = Complex.add(Complex.add(x0, x1), x2);
    data[1] = Complex.add(Complex.add(x0, Complex.mul(x1, w3)), Complex.mul(x2, w3_2));
    data[2] = Complex.add(Complex.add(x0, Complex.mul(x1, w3_2)), Complex.mul(x2, w3));
}

/// DFT of size 4 (scalar version).
fn dft4InPlaceScalar(data: []Complex, inv: bool) void {
    if (data.len != 4) return;

    const a0 = data[0];
    const a1 = data[1];
    const a2 = data[2];
    const a3 = data[3];

    const b0 = Complex.add(a0, a2);
    const b2 = Complex.sub(a0, a2);
    const b1 = Complex.add(a1, a3);
    const b3 = Complex.sub(a1, a3);

    // -i * (re + im*i) = im - re*i for forward, +i for inverse
    const sign: f64 = if (inv) 1.0 else -1.0;
    const b3_tw = Complex.init(-b3.im * sign, b3.re * sign);

    data[0] = Complex.add(b0, b1);
    data[1] = Complex.add(b2, b3_tw);
    data[2] = Complex.sub(b0, b1);
    data[3] = Complex.sub(b2, b3_tw);
}

/// DFT of size 5 (scalar version).
fn dft5InPlaceScalar(data: []Complex, inv: bool) void {
    if (data.len != 5) return;
    const sign: f64 = if (inv) 1.0 else -1.0;

    const c1: f64 = 0.30901699437494742;
    const s1: f64 = 0.95105651629515353;
    const c2: f64 = -0.80901699437494742;
    const s2: f64 = 0.58778525229247313;

    const w5_1 = Complex.init(c1, sign * s1);
    const w5_2 = Complex.init(c2, sign * s2);
    const w5_3 = Complex.init(c2, -sign * s2);
    const w5_4 = Complex.init(c1, -sign * s1);

    const x0 = data[0];
    const x1 = data[1];
    const x2 = data[2];
    const x3 = data[3];
    const x4 = data[4];

    data[0] = Complex.add(Complex.add(Complex.add(Complex.add(x0, x1), x2), x3), x4);
    data[1] = Complex.add(Complex.add(Complex.add(Complex.add(x0, Complex.mul(x1, w5_1)), Complex.mul(x2, w5_2)), Complex.mul(x3, w5_3)), Complex.mul(x4, w5_4));
    data[2] = Complex.add(Complex.add(Complex.add(Complex.add(x0, Complex.mul(x1, w5_2)), Complex.mul(x2, w5_4)), Complex.mul(x3, w5_1)), Complex.mul(x4, w5_3));
    data[3] = Complex.add(Complex.add(Complex.add(Complex.add(x0, Complex.mul(x1, w5_3)), Complex.mul(x2, w5_1)), Complex.mul(x3, w5_4)), Complex.mul(x4, w5_2));
    data[4] = Complex.add(Complex.add(Complex.add(Complex.add(x0, Complex.mul(x1, w5_4)), Complex.mul(x2, w5_3)), Complex.mul(x3, w5_2)), Complex.mul(x4, w5_1));
}

/// Mixed-radix FFT without normalization (scalar version for benchmarking).
pub fn mixedRadixFftNoNormScalar(data: []Complex, scratch: []Complex, inv: bool) void {
    const n = data.len;
    if (n <= 1) return;

    switch (n) {
        2 => dft2InPlaceScalar(data, inv),
        3 => dft3InPlaceScalar(data, inv),
        4 => dft4InPlaceScalar(data, inv),
        5 => dft5InPlaceScalar(data, inv),
        else => {
            var factor: usize = 0;
            for ([_]usize{ 2, 3, 5 }) |f| {
                if (n % f == 0) {
                    factor = f;
                    break;
                }
            }

            if (factor == 0) {
                dftDirect(data, scratch[0..n], inv);
                for (0..n) |i| {
                    data[i] = scratch[i];
                }
                return;
            }

            const N1 = factor;
            const N2 = n / factor;

            for (0..N1) |n1| {
                for (0..N2) |n2| {
                    scratch[n2] = data[n1 + N1 * n2];
                }
                mixedRadixFftNoNormScalar(scratch[0..N2], scratch[N2..], inv);
                for (0..N2) |n2| {
                    data[n1 + N1 * n2] = scratch[n2];
                }
            }

            const sign: f64 = if (inv) 1.0 else -1.0;
            const angle_base = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

            for (0..N1) |n1| {
                for (0..N2) |k2| {
                    if (n1 * k2 != 0) {
                        const tw = Complex.expi(angle_base * @as(f64, @floatFromInt(n1 * k2)));
                        const idx = n1 + N1 * k2;
                        data[idx] = Complex.mul(data[idx], tw);
                    }
                }
            }

            const output = scratch[n .. 2 * n];

            for (0..N2) |k2| {
                for (0..N1) |n1| {
                    scratch[n1] = data[n1 + N1 * k2];
                }
                mixedRadixFftNoNormScalar(scratch[0..N1], scratch[N1..n], inv);
                for (0..N1) |k1| {
                    output[N2 * k1 + k2] = scratch[k1];
                }
            }

            for (0..n) |i| {
                data[i] = output[i];
            }
        },
    }
}

// ============== Tests ==============

test "dft2 basic" {
    const input = [2]Complex{ Complex.init(1, 0), Complex.init(2, 0) };
    var output: [2]Complex = undefined;

    dft2(&input, &output, false);

    try std.testing.expectApproxEqAbs(output[0].re, 3.0, 1e-10); // 1+2
    try std.testing.expectApproxEqAbs(output[1].re, -1.0, 1e-10); // 1-2
}

test "dft3 vs direct" {
    const input = [3]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
    };

    var expected: [3]Complex = undefined;
    dftDirect(&input, &expected, false);

    var output: [3]Complex = undefined;
    dft3(&input, &output, false);

    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(output[i].re, expected[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(output[i].im, expected[i].im, 1e-10);
    }
}

test "fft6 vs direct - simple input" {
    // Simple input: [1, 0, 0, 0, 0, 0]
    // DFT should give [1, 1, 1, 1, 1, 1]
    var data = [6]Complex{
        Complex.init(1, 0),
        Complex.init(0, 0),
        Complex.init(0, 0),
        Complex.init(0, 0),
        Complex.init(0, 0),
        Complex.init(0, 0),
    };

    var expected: [6]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft6(&data, false);

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-10);
    }
}

test "fft6 vs direct - sequential input" {
    var data = [6]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
    };

    var expected: [6]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft6(&data, false);

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft6 vs direct - complex input" {
    var data = [6]Complex{
        Complex.init(1, 0),
        Complex.init(2, 1),
        Complex.init(3, 2),
        Complex.init(4, 3),
        Complex.init(5, 4),
        Complex.init(6, 5),
    };

    var expected: [6]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft6(&data, false);

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft6 roundtrip" {
    var data = [6]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
    };
    const original = data;

    fft6(&data, false);
    fft6(&data, true);

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "dft5 vs direct" {
    const input = [5]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
    };

    var expected: [5]Complex = undefined;
    dftDirect(&input, &expected, false);

    var output: [5]Complex = undefined;
    dft5(&input, &output, false);

    for (0..5) |i| {
        try std.testing.expectApproxEqAbs(output[i].re, expected[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(output[i].im, expected[i].im, 1e-10);
    }
}

test "fft9 vs direct" {
    var data = [9]Complex{
        Complex.init(1, 0),
        Complex.init(2, 1),
        Complex.init(3, 2),
        Complex.init(4, 3),
        Complex.init(5, 4),
        Complex.init(6, 5),
        Complex.init(7, 6),
        Complex.init(8, 7),
        Complex.init(9, 8),
    };

    var expected: [9]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft9(&data, false);

    for (0..9) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft9 roundtrip" {
    var data = [9]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
        Complex.init(7, 0),
        Complex.init(8, 0),
        Complex.init(9, 0),
    };
    const original = data;

    fft9(&data, false);
    fft9(&data, true);

    for (0..9) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "fft10 vs direct" {
    var data = [10]Complex{
        Complex.init(1, 0),
        Complex.init(2, 1),
        Complex.init(3, 2),
        Complex.init(4, 3),
        Complex.init(5, 4),
        Complex.init(6, 5),
        Complex.init(7, 6),
        Complex.init(8, 7),
        Complex.init(9, 8),
        Complex.init(10, 9),
    };

    var expected: [10]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft10(&data, false);

    for (0..10) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft10 roundtrip" {
    var data = [10]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
        Complex.init(7, 0),
        Complex.init(8, 0),
        Complex.init(9, 0),
        Complex.init(10, 0),
    };
    const original = data;

    fft10(&data, false);
    fft10(&data, true);

    for (0..10) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "dft4 vs direct" {
    var data = [4]Complex{
        Complex.init(1, 0),
        Complex.init(2, 1),
        Complex.init(3, 2),
        Complex.init(4, 3),
    };

    var expected: [4]Complex = undefined;
    dftDirect(&data, &expected, false);

    dft4InPlace(&data, false);

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-10);
    }
}

test "fft12 vs direct" {
    var data = [12]Complex{
        Complex.init(1, 0),
        Complex.init(2, 1),
        Complex.init(3, 2),
        Complex.init(4, 3),
        Complex.init(5, 4),
        Complex.init(6, 5),
        Complex.init(7, 6),
        Complex.init(8, 7),
        Complex.init(9, 8),
        Complex.init(10, 9),
        Complex.init(11, 10),
        Complex.init(12, 11),
    };

    var expected: [12]Complex = undefined;
    dftDirect(&data, &expected, false);

    fft12(&data, false);

    for (0..12) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "fft12 roundtrip" {
    var data = [12]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
        Complex.init(7, 0),
        Complex.init(8, 0),
        Complex.init(9, 0),
        Complex.init(10, 0),
        Complex.init(11, 0),
        Complex.init(12, 0),
    };
    const original = data;

    fft12(&data, false);
    fft12(&data, true);

    for (0..12) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "mixedRadixFftAlloc N=8" {
    // 8 = 2³ - simplest composite for debugging
    const allocator = std.testing.allocator;
    var data = [8]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
        Complex.init(5, 0),
        Complex.init(6, 0),
        Complex.init(7, 0),
        Complex.init(8, 0),
    };

    var expected: [8]Complex = undefined;
    dftDirect(&data, &expected, false);

    try mixedRadixFftAlloc(allocator, &data, false);

    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "mixedRadixFftAlloc N=24" {
    const allocator = std.testing.allocator;
    var data: [24]Complex = undefined;
    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt(i % 5));
    }

    var expected: [24]Complex = undefined;
    dftDirect(&data, &expected, false);

    try mixedRadixFftAlloc(allocator, &data, false);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-8);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-8);
    }
}

test "mixedRadixFftAlloc N=24 roundtrip" {
    const allocator = std.testing.allocator;
    var data: [24]Complex = undefined;
    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), 0);
    }
    const original = data;

    try mixedRadixFftAlloc(allocator, &data, false);
    try mixedRadixFftAlloc(allocator, &data, true);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "mixedRadixFftAlloc N=30" {
    // 30 = 2 × 3 × 5
    const allocator = std.testing.allocator;
    var data: [30]Complex = undefined;
    for (0..30) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt((i * 3) % 7));
    }

    var expected: [30]Complex = undefined;
    dftDirect(&data, &expected, false);

    try mixedRadixFftAlloc(allocator, &data, false);

    for (0..30) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-8);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-8);
    }
}

test "mixedRadixFftAlloc N=60" {
    // 60 = 2² × 3 × 5
    const allocator = std.testing.allocator;
    var data: [60]Complex = undefined;
    for (0..60) |i| {
        data[i] = Complex.init(@floatFromInt(i % 10), @floatFromInt(i % 7));
    }

    var expected: [60]Complex = undefined;
    dftDirect(&data, &expected, false);

    try mixedRadixFftAlloc(allocator, &data, false);

    for (0..60) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-7);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-7);
    }
}
