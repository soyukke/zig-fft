const std = @import("std");

/// Complex number with f64 components.
pub const Complex = struct {
    re: f64,
    im: f64,

    pub const zero = Complex{ .re = 0, .im = 0 };
    pub const one = Complex{ .re = 1, .im = 0 };
    pub const i = Complex{ .re = 0, .im = 1 };

    pub fn init(re: f64, im: f64) Complex {
        return .{ .re = re, .im = im };
    }

    pub fn add(a: Complex, b: Complex) Complex {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }

    pub fn sub(a: Complex, b: Complex) Complex {
        return .{ .re = a.re - b.re, .im = a.im - b.im };
    }

    pub fn mul(a: Complex, b: Complex) Complex {
        return .{
            .re = a.re * b.re - a.im * b.im,
            .im = a.re * b.im + a.im * b.re,
        };
    }

    pub fn scale(a: Complex, s: f64) Complex {
        return .{ .re = a.re * s, .im = a.im * s };
    }

    pub fn conj(a: Complex) Complex {
        return .{ .re = a.re, .im = -a.im };
    }

    pub fn abs(a: Complex) f64 {
        return @sqrt(a.re * a.re + a.im * a.im);
    }

    pub fn abs2(a: Complex) f64 {
        return a.re * a.re + a.im * a.im;
    }

    /// e^(i*theta)
    pub fn expi(theta: f64) Complex {
        return .{ .re = @cos(theta), .im = @sin(theta) };
    }

    /// e^z where z is complex
    pub fn exp(z: Complex) Complex {
        const r = @exp(z.re);
        return .{ .re = r * @cos(z.im), .im = r * @sin(z.im) };
    }

    pub fn eql(a: Complex, b: Complex, tol: f64) bool {
        return @abs(a.re - b.re) < tol and @abs(a.im - b.im) < tol;
    }

    pub fn format(
        self: Complex,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        if (self.im >= 0) {
            try writer.print("{d:.6}+{d:.6}i", .{ self.re, self.im });
        } else {
            try writer.print("{d:.6}{d:.6}i", .{ self.re, self.im });
        }
    }
};

// ============== Tests ==============

test "complex basic operations" {
    const a = Complex.init(1, 2);
    const b = Complex.init(3, 4);

    // add
    const sum = Complex.add(a, b);
    try std.testing.expectApproxEqAbs(sum.re, 4.0, 1e-10);
    try std.testing.expectApproxEqAbs(sum.im, 6.0, 1e-10);

    // sub
    const diff = Complex.sub(a, b);
    try std.testing.expectApproxEqAbs(diff.re, -2.0, 1e-10);
    try std.testing.expectApproxEqAbs(diff.im, -2.0, 1e-10);

    // mul: (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
    const prod = Complex.mul(a, b);
    try std.testing.expectApproxEqAbs(prod.re, -5.0, 1e-10);
    try std.testing.expectApproxEqAbs(prod.im, 10.0, 1e-10);

    // scale
    const scaled = Complex.scale(a, 2.0);
    try std.testing.expectApproxEqAbs(scaled.re, 2.0, 1e-10);
    try std.testing.expectApproxEqAbs(scaled.im, 4.0, 1e-10);

    // conj
    const conjugate = Complex.conj(a);
    try std.testing.expectApproxEqAbs(conjugate.re, 1.0, 1e-10);
    try std.testing.expectApproxEqAbs(conjugate.im, -2.0, 1e-10);

    // abs: |1+2i| = sqrt(5)
    try std.testing.expectApproxEqAbs(Complex.abs(a), @sqrt(5.0), 1e-10);
}

test "complex expi" {
    // e^(i*0) = 1
    const e0 = Complex.expi(0);
    try std.testing.expectApproxEqAbs(e0.re, 1.0, 1e-10);
    try std.testing.expectApproxEqAbs(e0.im, 0.0, 1e-10);

    // e^(i*pi/2) = i
    const e90 = Complex.expi(std.math.pi / 2.0);
    try std.testing.expectApproxEqAbs(e90.re, 0.0, 1e-10);
    try std.testing.expectApproxEqAbs(e90.im, 1.0, 1e-10);

    // e^(i*pi) = -1
    const e180 = Complex.expi(std.math.pi);
    try std.testing.expectApproxEqAbs(e180.re, -1.0, 1e-10);
    try std.testing.expectApproxEqAbs(e180.im, 0.0, 1e-10);
}
