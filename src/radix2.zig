const std = @import("std");
const Complex = @import("complex.zig").Complex;

/// Check if value is power of two.
pub fn isPowerOfTwo(n: usize) bool {
    return n != 0 and (n & (n - 1)) == 0;
}

/// In-place radix-2 FFT for complex data.
/// Requires n to be a power of two.
pub fn fft1d(data: []Complex, inverse: bool) void {
    const n = data.len;
    if (n <= 1) return;

    // Bit-reversal permutation
    var j: usize = 0;
    var idx: usize = 1;
    while (idx < n) : (idx += 1) {
        var bit = n >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (idx < j) {
            const tmp = data[idx];
            data[idx] = data[j];
            data[j] = tmp;
        }
    }

    // Cooley-Tukey iterative FFT
    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const sign: f64 = if (inverse) 2.0 else -2.0;
        const angle = sign * std.math.pi / @as(f64, @floatFromInt(len));
        const wlen = Complex.expi(angle);
        var start: usize = 0;
        while (start < n) : (start += len) {
            var w = Complex.one;
            var j0: usize = 0;
            const half = len >> 1;
            while (j0 < half) : (j0 += 1) {
                const u = data[start + j0];
                const v = Complex.mul(data[start + j0 + half], w);
                data[start + j0] = Complex.add(u, v);
                data[start + j0 + half] = Complex.sub(u, v);
                w = Complex.mul(w, wlen);
            }
        }
    }

    // Normalize for inverse transform
    if (inverse) {
        const inv_n = 1.0 / @as(f64, @floatFromInt(n));
        for (data) |*v| {
            v.* = Complex.scale(v.*, inv_n);
        }
    }
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

/// Pre-computed plan for radix-2 FFT.
pub const Plan = struct {
    n: usize,
    bitrev: []usize,
    stage_offsets: []usize,
    stage_half: []usize,
    twiddles: []Complex,
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

        const twiddles = try allocator.alloc(Complex, total_twiddles);
        errdefer allocator.free(twiddles);

        s = 0;
        while (s < stage_count) : (s += 1) {
            const len = @as(usize, 1) << @intCast(s + 1);
            const half = stage_half[s];
            const base = stage_offsets[s];
            var k: usize = 0;
            while (k < half) : (k += 1) {
                const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(len));
                twiddles[base + k] = Complex.expi(angle);
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
            .stage_offsets = stage_offsets,
            .stage_half = stage_half,
            .twiddles = twiddles,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Plan) void {
        if (self.bitrev.len > 0) self.allocator.free(self.bitrev);
        if (self.stage_offsets.len > 0) self.allocator.free(self.stage_offsets);
        if (self.stage_half.len > 0) self.allocator.free(self.stage_half);
        if (self.twiddles.len > 0) self.allocator.free(self.twiddles);
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
        var idx: usize = 0;
        while (idx < n) : (idx += 1) {
            const j = self.bitrev[idx];
            if (idx < j) {
                const tmp = data[idx];
                data[idx] = data[j];
                data[j] = tmp;
            }
        }

        // Butterfly operations
        var s: usize = 0;
        while (s < self.stage_offsets.len) : (s += 1) {
            const half = self.stage_half[s];
            const len = half << 1;
            const base = self.stage_offsets[s];
            var start: usize = 0;
            while (start < n) : (start += len) {
                var j0: usize = 0;
                while (j0 < half) : (j0 += 1) {
                    var w = self.twiddles[base + j0];
                    if (inv) w = Complex.conj(w);
                    const u = data[start + j0];
                    const v = Complex.mul(data[start + j0 + half], w);
                    data[start + j0] = Complex.add(u, v);
                    data[start + j0 + half] = Complex.sub(u, v);
                }
            }
        }

        // Normalize for inverse
        if (inv) {
            const inv_n = 1.0 / @as(f64, @floatFromInt(n));
            for (data) |*v| {
                v.* = Complex.scale(v.*, inv_n);
            }
        }
    }
};

// ============== Tests ==============

test "isPowerOfTwo" {
    try std.testing.expect(isPowerOfTwo(1));
    try std.testing.expect(isPowerOfTwo(2));
    try std.testing.expect(isPowerOfTwo(4));
    try std.testing.expect(isPowerOfTwo(8));
    try std.testing.expect(isPowerOfTwo(1024));

    try std.testing.expect(!isPowerOfTwo(0));
    try std.testing.expect(!isPowerOfTwo(3));
    try std.testing.expect(!isPowerOfTwo(5));
    try std.testing.expect(!isPowerOfTwo(6));
    try std.testing.expect(!isPowerOfTwo(24));
}

test "radix2 fft1d basic" {
    var data = [_]Complex{
        Complex.init(1, 0),
        Complex.init(2, 0),
        Complex.init(3, 0),
        Complex.init(4, 0),
    };

    // Forward FFT
    fft1d(&data, false);

    // DFT of [1,2,3,4]:
    // X[0] = 1+2+3+4 = 10
    // X[1] = 1 + 2*e^(-iπ/2) + 3*e^(-iπ) + 4*e^(-i3π/2) = 1 - 2i - 3 + 4i = -2 + 2i
    // X[2] = 1 + 2*e^(-iπ) + 3*e^(-2iπ) + 4*e^(-3iπ) = 1 - 2 + 3 - 4 = -2
    // X[3] = 1 + 2*e^(-i3π/2) + 3*e^(-3iπ) + 4*e^(-i9π/2) = 1 + 2i - 3 - 4i = -2 - 2i
    try std.testing.expectApproxEqAbs(data[0].re, 10.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[0].im, 0.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[1].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[1].im, 2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[2].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[2].im, 0.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[3].re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[3].im, -2.0, 1e-10);

    // Inverse FFT should recover original
    fft1d(&data, true);
    try std.testing.expectApproxEqAbs(data[0].re, 1.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[1].re, 2.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[2].re, 3.0, 1e-10);
    try std.testing.expectApproxEqAbs(data[3].re, 4.0, 1e-10);
}

test "radix2 Plan" {
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

    // Should recover original
    for (data, original) |d, o| {
        try std.testing.expectApproxEqAbs(d.re, o.re, 1e-10);
        try std.testing.expectApproxEqAbs(d.im, o.im, 1e-10);
    }
}
