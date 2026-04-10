//! Real FFT (RFFT) - Optimized FFT for real-valued input
//!
//! For real input of length N, the output has Hermitian symmetry:
//!   F[-k] = conj(F[k])
//! So we only need to store N/2+1 complex values.
//!
//! Algorithm:
//! 1. Pack N real values as N/2 complex values: z[k] = x[2k] + i*x[2k+1]
//! 2. Perform N/2-point complex FFT
//! 3. Unpack to get N/2+1 complex output values
//!
//! This gives ~2x speedup over complex FFT for real data.

const std = @import("std");
const Complex = @import("complex.zig").Complex;
const fft = @import("fft.zig");

/// Real FFT Plan for power-of-two sizes
pub const RealPlan = struct {
    n: usize, // Real input size (must be even)
    half_plan: fft.Plan1d, // N/2 complex FFT plan
    twiddles: []Complex, // Twiddle factors for unpack step
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !RealPlan {
        if (n == 0 or n % 2 != 0) return error.InvalidSize;

        const half_n = n / 2;
        var half_plan = try fft.Plan1d.init(allocator, half_n);
        errdefer half_plan.deinit();

        // Precompute twiddle factors: W_N^k = exp(-2πik/N)
        const twiddles = try allocator.alloc(Complex, half_n);
        errdefer allocator.free(twiddles);

        for (0..half_n) |k| {
            const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n));
            twiddles[k] = Complex.init(@cos(angle), @sin(angle));
        }

        return .{
            .n = n,
            .half_plan = half_plan,
            .twiddles = twiddles,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RealPlan) void {
        self.allocator.free(self.twiddles);
        self.half_plan.deinit();
    }

    /// Forward RFFT: real[N] -> complex[N/2+1]
    /// Input: real data of length N
    /// Output: complex data of length N/2+1 (only non-redundant coefficients)
    pub fn forward(self: *const RealPlan, real_input: []const f64, complex_output: []Complex) void {
        const n = self.n;
        const half_n = n / 2;

        if (real_input.len != n) return;
        if (complex_output.len != half_n + 1) return;

        // Step 1: Pack real data as complex
        // z[k] = x[2k] + i*x[2k+1]
        for (0..half_n) |k| {
            complex_output[k] = Complex.init(real_input[2 * k], real_input[2 * k + 1]);
        }

        // Step 2: N/2-point complex FFT in-place
        self.half_plan.forward(complex_output[0..half_n]);

        // Step 3: Unpack to get full spectrum
        // Z[k] = FFT(z) where z[m] = x[2m] + i*x[2m+1]
        // We have: Z[k] = E[k] + i*O[k]
        //   where E[k] = sum_{m=0}^{N/2-1} x[2m] * W_{N/2}^{mk}
        //         O[k] = sum_{m=0}^{N/2-1} x[2m+1] * W_{N/2}^{mk}
        //
        // The full DFT X[k] = E'[k] + W_N^k * O'[k]
        //   where E'[k] = sum_{m=0}^{N/2-1} x[2m] * W_{N/2}^{mk} = E[k]
        //         O'[k] = sum_{m=0}^{N/2-1} x[2m+1] * W_{N/2}^{mk} = O[k]
        //
        // From Z[k] and Z[N/2-k], we extract E[k] and O[k]:
        //   Z[k] = E[k] + i*O[k]
        //   Z[N/2-k]* = E[k] - i*O[k]  (using E[-k]=E[k]*, O[-k]=O[k]* for real data)
        //   E[k] = (Z[k] + Z[N/2-k]*) / 2
        //   O[k] = (Z[k] - Z[N/2-k]*) / (2i)

        // Save Z values before overwriting (process pairs symmetrically)
        const z0 = complex_output[0];

        // X[0] is purely real: X[0] = E[0] + O[0]
        // E[0] = Re(Z[0]), O[0] = Im(Z[0])
        complex_output[0] = Complex.init(z0.re + z0.im, 0);

        // X[N/2] is also purely real: X[N/2] = E[0] - O[0]
        complex_output[half_n] = Complex.init(z0.re - z0.im, 0);

        // Process pairs (k, N/2-k) together to avoid overwrite issues
        // For k = 1 to N/4, compute both X[k] and X[N/2-k]
        var k: usize = 1;
        while (k < half_n - k) : (k += 1) {
            const zk = complex_output[k];
            const zn_k = complex_output[half_n - k];

            // E[k] = (Z[k] + conj(Z[N/2-k])) / 2
            const e_k = Complex.init(
                (zk.re + zn_k.re) / 2.0,
                (zk.im - zn_k.im) / 2.0,
            );

            // O[k] = (Z[k] - conj(Z[N/2-k])) / (2i)
            // = -i * (Z[k] - conj(Z[N/2-k])) / 2
            // If diff = Z[k] - conj(Z[N/2-k]) = (a + bi), then diff/(2i) = (b - ai)/2
            const diff_re = (zk.re - zn_k.re) / 2.0;
            const diff_im = (zk.im + zn_k.im) / 2.0;
            const o_k = Complex.init(diff_im, -diff_re);

            // X[k] = E[k] + W_N^k * O[k]
            const wk = self.twiddles[k];
            const w_o_k = Complex.mul(wk, o_k);
            complex_output[k] = Complex.add(e_k, w_o_k);

            // E[N/2-k] = conj(E[k]) for real input
            // O[N/2-k] = conj(O[k]) for real input
            // X[N/2-k] = E[N/2-k] + W_N^{N/2-k} * O[N/2-k]
            //          = conj(E[k]) + W_N^{N/2-k} * conj(O[k])
            const e_n_k = Complex.conj(e_k);
            const o_n_k = Complex.conj(o_k);
            const wn_k = self.twiddles[half_n - k];
            const w_o_n_k = Complex.mul(wn_k, o_n_k);
            complex_output[half_n - k] = Complex.add(e_n_k, w_o_n_k);
        }

        // Handle middle element if N/2 is even (k = N/4)
        if (half_n % 2 == 0) {
            const mid = half_n / 2;
            const z_mid = complex_output[mid];
            // E[mid] = Re(Z[mid]), O[mid] = Im(Z[mid]) (since Z[N/2-mid] = Z[mid])
            const e_mid = Complex.init(z_mid.re, 0);
            const o_mid = Complex.init(z_mid.im, 0);
            const w_mid = self.twiddles[mid];
            const w_o_mid = Complex.mul(w_mid, o_mid);
            complex_output[mid] = Complex.add(e_mid, w_o_mid);
        }
    }

    /// Inverse RFFT: complex[N/2+1] -> real[N]
    /// Input: complex data of length N/2+1 (Hermitian symmetric spectrum)
    /// Output: real data of length N
    /// Note: Uses real_output as scratch space during computation
    pub fn inverse(self: *const RealPlan, complex_input: []const Complex, real_output: []f64) void {
        const n = self.n;
        const half_n = n / 2;

        if (complex_input.len != half_n + 1) return;
        if (real_output.len != n) return;

        // Use real_output as Complex scratch (reinterpret)
        const scratch: [*]Complex = @ptrCast(@alignCast(real_output.ptr));
        const z = scratch[0..half_n];

        // Step 1: Pack spectrum back to Z[k] = E[k] + i*O[k]
        // From X[k] and X[N/2-k], recover E[k] and O[k]
        //
        // X[k] = E[k] + W_N^k * O[k]
        // X[N-k]* = X[k] for real signal, so X[N/2-k]* relates to E and O
        //
        // Inverse of forward unpack:
        // E[k] = (X[k] + X[N/2-k]*) / 2  (for k != 0, N/2)
        // O[k] = (X[k] - X[N/2-k]*) / (2 * W_N^k)
        // Z[k] = E[k] + i*O[k]

        const x0 = complex_input[0];
        const x_half = complex_input[half_n];

        // Z[0] from X[0] and X[N/2]
        // E[0] = (X[0] + X[N/2]) / 2, O[0] = (X[0] - X[N/2]) / 2
        // Z[0] = E[0] + i*O[0]
        z[0] = Complex.init((x0.re + x_half.re) / 2.0, (x0.re - x_half.re) / 2.0);

        // Process pairs
        var k: usize = 1;
        while (k < half_n - k) : (k += 1) {
            const xk = complex_input[k];
            const xn_k = complex_input[half_n - k];

            // E[k] = (X[k] + conj(X[N/2-k])) / 2
            const e_k = Complex.init(
                (xk.re + xn_k.re) / 2.0,
                (xk.im - xn_k.im) / 2.0,
            );

            // W_N^k * O[k] = (X[k] - conj(X[N/2-k])) / 2
            // O[k] = conj(W_N^k) * (X[k] - conj(X[N/2-k])) / 2
            const diff = Complex.init(
                (xk.re - xn_k.re) / 2.0,
                (xk.im + xn_k.im) / 2.0,
            );
            const wk_conj = Complex.conj(self.twiddles[k]);
            const o_k = Complex.mul(wk_conj, diff);

            // Z[k] = E[k] + i*O[k] = E[k].re - O[k].im + i*(E[k].im + O[k].re)
            z[k] = Complex.init(e_k.re - o_k.im, e_k.im + o_k.re);

            // Z[N/2-k] = E[N/2-k] + i*O[N/2-k] = conj(E[k]) + i*conj(O[k])
            const e_n_k = Complex.conj(e_k);
            const o_n_k = Complex.conj(o_k);
            z[half_n - k] = Complex.init(e_n_k.re - o_n_k.im, e_n_k.im + o_n_k.re);
        }

        // Handle middle element if half_n is even
        if (half_n % 2 == 0) {
            const mid = half_n / 2;
            const x_mid = complex_input[mid];
            // For the middle element, X[mid] = E[mid] + W_N^{mid} * O[mid]
            // and X[N/2-mid] = X[mid], so E[mid] is real, O[mid] is real
            // E[mid] = Re(X[mid]), W_N^{mid}*O[mid] = Im(X[mid])*i (since W_N^{N/4} = -i for N/2 even)
            // Actually W_N^{N/4} = exp(-i*pi/2) = -i
            // So O[mid] = -Im(X[mid]) * i / (-i) = Im(X[mid])... let me recalculate
            // W_N^{mid} where mid = N/4 and W_N = exp(-2pi*i/N)
            // W_N^{N/4} = exp(-2pi*i*(N/4)/N) = exp(-i*pi/2) = -i
            // So W_N^{mid} * O[mid] = -i * O[mid]
            // If X[mid] = E[mid] + (-i)*O[mid] = E[mid].re - O[mid].im + i*(E[mid].im - O[mid].re)
            // For Hermitian: E[mid] and O[mid] are both real
            // X[mid].re = E[mid], X[mid].im = -O[mid]
            // So E[mid] = X[mid].re, O[mid] = -X[mid].im
            const e_mid = x_mid.re;
            const o_mid = -x_mid.im;
            z[mid] = Complex.init(e_mid, o_mid);
        }

        // Step 2: Inverse N/2-point complex FFT
        self.half_plan.inverse(z);

        // Step 3: Unpack to real
        // z[k] = x[2k] + i*x[2k+1]
        // After IFFT, z contains interleaved real values
        // real_output is already pointing to the same memory as z
        // Just need to ensure the layout is correct (it should be)
    }
};

// ============== Tests ==============

test "RealPlan forward basic" {
    const allocator = std.testing.allocator;

    var plan = try RealPlan.init(allocator, 8);
    defer plan.deinit();

    // Test with simple input: [1, 0, 0, 0, 0, 0, 0, 0]
    // DFT should give all 1s
    const real_input = [_]f64{ 1, 0, 0, 0, 0, 0, 0, 0 };
    var complex_output: [5]Complex = undefined;

    plan.forward(&real_input, &complex_output);

    // X[k] should all be 1 + 0i
    for (0..5) |k| {
        try std.testing.expectApproxEqAbs(complex_output[k].re, 1.0, 1e-10);
        try std.testing.expectApproxEqAbs(complex_output[k].im, 0.0, 1e-10);
    }
}

test "RealPlan forward vs complex FFT" {
    const allocator = std.testing.allocator;

    const n = 8;
    var rfft_plan = try RealPlan.init(allocator, n);
    defer rfft_plan.deinit();

    var cfft_plan = try fft.Plan1d.init(allocator, n);
    defer cfft_plan.deinit();

    // Test with arbitrary real input
    const real_input = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // RFFT
    var rfft_output: [5]Complex = undefined;
    rfft_plan.forward(&real_input, &rfft_output);

    // Complex FFT (convert real to complex first)
    var cfft_data: [8]Complex = undefined;
    for (0..8) |i| {
        cfft_data[i] = Complex.init(real_input[i], 0);
    }
    cfft_plan.forward(&cfft_data);

    // Compare first N/2+1 values
    for (0..5) |k| {
        try std.testing.expectApproxEqAbs(rfft_output[k].re, cfft_data[k].re, 1e-10);
        try std.testing.expectApproxEqAbs(rfft_output[k].im, cfft_data[k].im, 1e-10);
    }
}

test "RealPlan size 16" {
    const allocator = std.testing.allocator;

    const n = 16;
    var rfft_plan = try RealPlan.init(allocator, n);
    defer rfft_plan.deinit();

    var cfft_plan = try fft.Plan1d.init(allocator, n);
    defer cfft_plan.deinit();

    // Test with arbitrary real input
    var real_input: [16]f64 = undefined;
    for (0..16) |i| {
        real_input[i] = @floatFromInt(i % 5);
    }

    // RFFT
    var rfft_output: [9]Complex = undefined;
    rfft_plan.forward(&real_input, &rfft_output);

    // Complex FFT
    var cfft_data: [16]Complex = undefined;
    for (0..16) |i| {
        cfft_data[i] = Complex.init(real_input[i], 0);
    }
    cfft_plan.forward(&cfft_data);

    // Compare
    for (0..9) |k| {
        try std.testing.expectApproxEqAbs(rfft_output[k].re, cfft_data[k].re, 1e-10);
        try std.testing.expectApproxEqAbs(rfft_output[k].im, cfft_data[k].im, 1e-10);
    }
}

test "RealPlan roundtrip" {
    const allocator = std.testing.allocator;

    const n = 8;
    var plan = try RealPlan.init(allocator, n);
    defer plan.deinit();

    const original = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var spectrum: [5]Complex = undefined;
    var recovered: [8]f64 = undefined;

    plan.forward(&original, &spectrum);
    plan.inverse(&spectrum, &recovered);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(recovered[i], original[i], 1e-10);
    }
}

test "RealPlan roundtrip size 16" {
    const allocator = std.testing.allocator;

    const n = 16;
    var plan = try RealPlan.init(allocator, n);
    defer plan.deinit();

    var original: [16]f64 = undefined;
    for (0..16) |i| {
        original[i] = @floatFromInt(i * i % 17);
    }

    var spectrum: [9]Complex = undefined;
    var recovered: [16]f64 = undefined;

    plan.forward(&original, &spectrum);
    plan.inverse(&spectrum, &recovered);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(recovered[i], original[i], 1e-10);
    }
}
