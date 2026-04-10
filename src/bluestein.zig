//! Bluestein's FFT Algorithm (Chirp-Z Transform)
//!
//! Computes DFT of arbitrary size N by converting to convolution,
//! which is computed using radix-2 FFT of size M >= 2N-1.
//!
//! Algorithm:
//!   X[k] = Σ(n=0..N-1) x[n] * W_N^(nk)  where W_N = e^(-2πi/N)
//!
//!   Using Bluestein's identity: nk = (n² + k² - (k-n)²) / 2
//!
//!   X[k] = W_N^(k²/2) * Σ(n=0..N-1) [x[n] * W_N^(n²/2)] * [W_N^(-(k-n)²/2)]
//!
//!   This is a convolution that can be computed via FFT.

const std = @import("std");
const Complex = @import("complex.zig").Complex;
const radix2 = @import("radix2.zig");

/// Find the smallest power of 2 >= n.
fn nextPowerOfTwo(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

/// Bluestein FFT Plan for arbitrary size N.
pub const Plan = struct {
    n: usize, // Original size
    m: usize, // Padded size (power of 2, >= 2N-1)
    chirp: []Complex, // W_N^(n²/2) for n = 0..N-1
    chirp_conj_fft: []Complex, // FFT of zero-padded conjugate chirp
    radix2_plan: radix2.Plan, // Plan for size-M FFT
    work_a: []Complex, // Work buffer for input sequence
    work_b: []Complex, // Work buffer for convolution
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !Plan {
        if (n == 0) return error.InvalidSize;

        // M must be >= 2N-1 and a power of 2
        const m = nextPowerOfTwo(2 * n - 1);

        // Allocate chirp factors: W_N^(n²/2) for n = 0..N-1
        const chirp = try allocator.alloc(Complex, n);
        errdefer allocator.free(chirp);

        const w_n = -std.math.pi / @as(f64, @floatFromInt(n)); // -π/N (for W_N^(1/2))
        for (0..n) |i| {
            const i_f = @as(f64, @floatFromInt(i));
            // W_N^(n²/2) = e^(-iπn²/N)
            chirp[i] = Complex.expi(w_n * i_f * i_f);
        }

        // Create chirp_conj_fft: zero-padded FFT of W_N^(-n²/2)
        // Index mapping: b[k-n] where k-n ranges from -(N-1) to N-1
        // We use circular indexing: negative indices wrap around
        const chirp_conj_fft = try allocator.alloc(Complex, m);
        errdefer allocator.free(chirp_conj_fft);

        // Initialize to zero
        for (chirp_conj_fft) |*c| {
            c.* = Complex.zero;
        }

        // Fill chirp_conj_fft:
        // b[j] = W_N^(-j²/2) for j = 0..N-1 (positive indices)
        // b[M-j] = W_N^(-j²/2) for j = 1..N-1 (negative indices wrap)
        for (0..n) |j| {
            const j_f = @as(f64, @floatFromInt(j));
            const val = Complex.expi(-w_n * j_f * j_f); // W_N^(-j²/2) = e^(iπj²/N)
            chirp_conj_fft[j] = val;
            if (j > 0 and j < n) {
                chirp_conj_fft[m - j] = val; // Wrap negative indices
            }
        }

        // Compute FFT of chirp_conj
        var radix2_plan = try radix2.Plan.init(allocator, m);
        errdefer radix2_plan.deinit();

        radix2_plan.forward(chirp_conj_fft);

        // Allocate work buffers
        const work_a = try allocator.alloc(Complex, m);
        errdefer allocator.free(work_a);

        const work_b = try allocator.alloc(Complex, m);
        errdefer allocator.free(work_b);

        return .{
            .n = n,
            .m = m,
            .chirp = chirp,
            .chirp_conj_fft = chirp_conj_fft,
            .radix2_plan = radix2_plan,
            .work_a = work_a,
            .work_b = work_b,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan) void {
        self.allocator.free(self.chirp);
        self.allocator.free(self.chirp_conj_fft);
        self.allocator.free(self.work_a);
        self.allocator.free(self.work_b);
        self.radix2_plan.deinit();
    }

    pub fn forward(self: *const Plan, data: []Complex) void {
        self.execute(data, false);
    }

    pub fn inverse(self: *const Plan, data: []Complex) void {
        self.execute(data, true);
    }

    fn execute(self: *const Plan, data: []Complex, inv: bool) void {
        const n = self.n;
        const m = self.m;

        if (data.len != n) return;

        // Step 1: Multiply input by chirp factors
        // a[n] = x[n] * W_N^(n²/2)  (or conjugate for inverse)
        for (self.work_a) |*a| {
            a.* = Complex.zero;
        }

        for (0..n) |i| {
            if (inv) {
                // For inverse: use conjugate chirp
                self.work_a[i] = Complex.mul(data[i], Complex.conj(self.chirp[i]));
            } else {
                self.work_a[i] = Complex.mul(data[i], self.chirp[i]);
            }
        }

        // Step 2: FFT of a
        self.radix2_plan.forward(self.work_a);

        // Step 3: Pointwise multiply with chirp_conj_fft
        // For inverse FFT, we need to use the conjugate of chirp_conj_fft
        for (0..m) |i| {
            if (inv) {
                self.work_b[i] = Complex.mul(self.work_a[i], Complex.conj(self.chirp_conj_fft[i]));
            } else {
                self.work_b[i] = Complex.mul(self.work_a[i], self.chirp_conj_fft[i]);
            }
        }

        // Step 4: Inverse FFT
        self.radix2_plan.inverse(self.work_b);

        // Step 5: Multiply by chirp and extract result
        // X[k] = W_N^(k²/2) * (convolution result at k)
        for (0..n) |k| {
            if (inv) {
                // For inverse: use conjugate chirp and normalize
                data[k] = Complex.mul(self.work_b[k], Complex.conj(self.chirp[k]));
                data[k] = Complex.scale(data[k], 1.0 / @as(f64, @floatFromInt(n)));
            } else {
                data[k] = Complex.mul(self.work_b[k], self.chirp[k]);
            }
        }
    }
};

/// Direct DFT computation for testing (O(N²)).
pub fn dftDirect(data: []const Complex, result: []Complex, inverse: bool) void {
    const n = data.len;
    if (result.len != n) return;

    const sign: f64 = if (inverse) 1.0 else -1.0;
    const w = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

    for (0..n) |k| {
        var sum = Complex.zero;
        for (0..n) |j| {
            const angle = w * @as(f64, @floatFromInt(k * j));
            const twiddle = Complex.expi(angle);
            sum = Complex.add(sum, Complex.mul(data[j], twiddle));
        }
        if (inverse) {
            result[k] = Complex.scale(sum, 1.0 / @as(f64, @floatFromInt(n)));
        } else {
            result[k] = sum;
        }
    }
}

// ============== Tests ==============

test "nextPowerOfTwo" {
    try std.testing.expectEqual(@as(usize, 1), nextPowerOfTwo(0));
    try std.testing.expectEqual(@as(usize, 1), nextPowerOfTwo(1));
    try std.testing.expectEqual(@as(usize, 2), nextPowerOfTwo(2));
    try std.testing.expectEqual(@as(usize, 4), nextPowerOfTwo(3));
    try std.testing.expectEqual(@as(usize, 4), nextPowerOfTwo(4));
    try std.testing.expectEqual(@as(usize, 8), nextPowerOfTwo(5));
    try std.testing.expectEqual(@as(usize, 8), nextPowerOfTwo(7));
    try std.testing.expectEqual(@as(usize, 8), nextPowerOfTwo(8));
    try std.testing.expectEqual(@as(usize, 16), nextPowerOfTwo(9));
    try std.testing.expectEqual(@as(usize, 32), nextPowerOfTwo(24));
}

test "dftDirect basic" {
    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
    };
    var result: [4]Complex = undefined;

    dftDirect(&data, &result, false);

    // DFT of [1,2,3,4]: X[0]=10, X[1]=-2+2i, X[2]=-2, X[3]=-2-2i
    try std.testing.expectApproxEqAbs(result[0].re, 10.0, 1e-10);
    try std.testing.expectApproxEqAbs(result[0].im, 0.0, 1e-10);
    try std.testing.expectApproxEqAbs(result[1].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(result[1].im, 2.0, 1e-10);
    try std.testing.expectApproxEqAbs(result[2].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(result[2].im, 0.0, 1e-10);
}

test "bluestein size 3" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 3);
    defer plan.deinit();

    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
    };

    // Compute expected result via direct DFT
    var expected: [3]Complex = undefined;
    dftDirect(&data, &expected, false);

    // Compute via Bluestein
    plan.forward(&data);

    // Compare
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-10);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-10);
    }
}

test "bluestein size 5" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 5);
    defer plan.deinit();

    var data = [_]Complex{
        Complex.init(1, 2),
        Complex.init(3, 4),
        Complex.init(5, 6),
        Complex.init(7, 8),
        Complex.init(9, 10),
    };

    var expected: [5]Complex = undefined;
    dftDirect(&data, &expected, false);

    plan.forward(&data);

    for (0..5) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "bluestein size 6 roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 6);
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

    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "bluestein size 7" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 7);
    defer plan.deinit();

    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 1),
        Complex.init(3, 2),
        Complex.init(4, 3),
        Complex.init(5, 4),
        Complex.init(6, 5),
        Complex.init(7, 6),
    };

    var expected: [7]Complex = undefined;
    dftDirect(&data, &expected, false);

    plan.forward(&data);

    for (0..7) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-9);
    }
}

test "bluestein size 24" {
    const allocator = std.testing.allocator;

    var plan = try Plan.init(allocator, 24);
    defer plan.deinit();

    var data: [24]Complex = undefined;
    var original: [24]Complex = undefined;

    for (0..24) |i| {
        data[i] = Complex.init(@floatFromInt(i), @floatFromInt(i * 2));
        original[i] = data[i];
    }

    var expected: [24]Complex = undefined;
    dftDirect(&original, &expected, false);

    plan.forward(&data);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, expected[i].re, 1e-8);
        try std.testing.expectApproxEqAbs(data[i].im, expected[i].im, 1e-8);
    }

    // Roundtrip test
    plan.inverse(&data);

    for (0..24) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-8);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-8);
    }
}
