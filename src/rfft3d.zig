//! 3D Real FFT for real-valued 3D data (e.g., electron density)
//!
//! For real input of size [nx, ny, nz], output has Hermitian symmetry along x-axis.
//! Output size: [nx/2+1, ny, nz] complex values.
//!
//! Algorithm:
//! Forward: RFFT along x -> FFT along y -> FFT along z
//! Inverse: IFFT along z -> IFFT along y -> IRFFT along x
//!
//! Memory layout (row-major, x fastest):
//!   real[ix + nx * (iy + ny * iz)]
//!   complex[ix + (nx/2+1) * (iy + ny * iz)]

const std = @import("std");
const Complex = @import("complex.zig").Complex;
const fft = @import("fft.zig");
const rfft = @import("rfft.zig");

/// 3D Real FFT Plan
pub const RealPlan3d = struct {
    nx: usize, // Real input x-size (must be even)
    ny: usize,
    nz: usize,
    nx_complex: usize, // = nx/2 + 1
    rfft_x: rfft.RealPlan, // RFFT for x-axis
    fft_y: fft.Plan1d, // FFT for y-axis
    fft_z: fft.Plan1d, // FFT for z-axis
    scratch_y: []Complex, // Scratch for y-axis transforms
    scratch_z: []Complex, // Scratch for z-axis transforms
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !RealPlan3d {
        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;
        if (nx % 2 != 0) return error.InvalidSize; // nx must be even for RFFT

        const nx_complex = nx / 2 + 1;

        var rfft_x = try rfft.RealPlan.init(allocator, nx);
        errdefer rfft_x.deinit();

        var fft_y = try fft.Plan1d.init(allocator, ny);
        errdefer fft_y.deinit();

        var fft_z = try fft.Plan1d.init(allocator, nz);
        errdefer fft_z.deinit();

        const scratch_y = try allocator.alloc(Complex, ny);
        errdefer allocator.free(scratch_y);

        const scratch_z = try allocator.alloc(Complex, nz);
        errdefer allocator.free(scratch_z);

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .nx_complex = nx_complex,
            .rfft_x = rfft_x,
            .fft_y = fft_y,
            .fft_z = fft_z,
            .scratch_y = scratch_y,
            .scratch_z = scratch_z,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RealPlan3d) void {
        self.allocator.free(self.scratch_y);
        self.allocator.free(self.scratch_z);
        self.rfft_x.deinit();
        self.fft_y.deinit();
        self.fft_z.deinit();
    }

    /// Forward 3D RFFT: real[nx*ny*nz] -> complex[(nx/2+1)*ny*nz]
    pub fn forward(self: *RealPlan3d, real_input: []const f64, complex_output: []Complex) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;
        const nx_c = self.nx_complex;

        if (real_input.len != nx * ny * nz) return;
        if (complex_output.len != nx_c * ny * nz) return;

        // Phase 1: RFFT along x-axis for each (y, z)
        for (0..nz) |iz| {
            for (0..ny) |iy| {
                const real_offset = nx * (iy + ny * iz);
                const complex_offset = nx_c * (iy + ny * iz);
                self.rfft_x.forward(
                    real_input[real_offset .. real_offset + nx],
                    complex_output[complex_offset .. complex_offset + nx_c],
                );
            }
        }

        // Phase 2: FFT along y-axis for each (x, z)
        for (0..nz) |iz| {
            for (0..nx_c) |ix| {
                // Gather y-slice
                for (0..ny) |iy| {
                    self.scratch_y[iy] = complex_output[ix + nx_c * (iy + ny * iz)];
                }
                // FFT
                self.fft_y.forward(self.scratch_y);
                // Scatter back
                for (0..ny) |iy| {
                    complex_output[ix + nx_c * (iy + ny * iz)] = self.scratch_y[iy];
                }
            }
        }

        // Phase 3: FFT along z-axis for each (x, y)
        for (0..ny) |iy| {
            for (0..nx_c) |ix| {
                // Gather z-slice
                for (0..nz) |iz| {
                    self.scratch_z[iz] = complex_output[ix + nx_c * (iy + ny * iz)];
                }
                // FFT
                self.fft_z.forward(self.scratch_z);
                // Scatter back
                for (0..nz) |iz| {
                    complex_output[ix + nx_c * (iy + ny * iz)] = self.scratch_z[iz];
                }
            }
        }
    }

    /// Inverse 3D RFFT: complex[(nx/2+1)*ny*nz] -> real[nx*ny*nz]
    /// Note: complex_input is modified during computation (used as scratch)
    pub fn inverse(self: *RealPlan3d, complex_input: []Complex, real_output: []f64) void {
        const nx = self.nx;
        const ny = self.ny;
        const nz = self.nz;
        const nx_c = self.nx_complex;

        if (complex_input.len != nx_c * ny * nz) return;
        if (real_output.len != nx * ny * nz) return;

        // Phase 1: IFFT along z-axis for each (x, y)
        for (0..ny) |iy| {
            for (0..nx_c) |ix| {
                // Gather z-slice
                for (0..nz) |iz| {
                    self.scratch_z[iz] = complex_input[ix + nx_c * (iy + ny * iz)];
                }
                // IFFT
                self.fft_z.inverse(self.scratch_z);
                // Scatter back
                for (0..nz) |iz| {
                    complex_input[ix + nx_c * (iy + ny * iz)] = self.scratch_z[iz];
                }
            }
        }

        // Phase 2: IFFT along y-axis for each (x, z)
        for (0..nz) |iz| {
            for (0..nx_c) |ix| {
                // Gather y-slice
                for (0..ny) |iy| {
                    self.scratch_y[iy] = complex_input[ix + nx_c * (iy + ny * iz)];
                }
                // IFFT
                self.fft_y.inverse(self.scratch_y);
                // Scatter back
                for (0..ny) |iy| {
                    complex_input[ix + nx_c * (iy + ny * iz)] = self.scratch_y[iy];
                }
            }
        }

        // Phase 3: IRFFT along x-axis for each (y, z)
        for (0..nz) |iz| {
            for (0..ny) |iy| {
                const complex_offset = nx_c * (iy + ny * iz);
                const real_offset = nx * (iy + ny * iz);
                self.rfft_x.inverse(
                    complex_input[complex_offset .. complex_offset + nx_c],
                    real_output[real_offset .. real_offset + nx],
                );
            }
        }
    }

    /// Get the complex output size
    pub fn complexSize(self: *const RealPlan3d) usize {
        return self.nx_complex * self.ny * self.nz;
    }

    /// Get the real input size
    pub fn realSize(self: *const RealPlan3d) usize {
        return self.nx * self.ny * self.nz;
    }
};

// ============== Tests ==============

test "RealPlan3d forward basic" {
    const allocator = std.testing.allocator;

    var plan = try RealPlan3d.init(allocator, 4, 4, 4);
    defer plan.deinit();

    // All ones input -> only DC component is non-zero
    var real_input: [64]f64 = undefined;
    for (0..64) |i| {
        real_input[i] = 1.0;
    }

    var complex_output: [3 * 4 * 4]Complex = undefined; // (4/2+1) * 4 * 4 = 48
    plan.forward(&real_input, &complex_output);

    // DC component (0,0,0) should be 64 (sum of all inputs)
    try std.testing.expectApproxEqAbs(complex_output[0].re, 64.0, 1e-10);
    try std.testing.expectApproxEqAbs(complex_output[0].im, 0.0, 1e-10);

    // All other components should be ~0
    for (1..48) |i| {
        try std.testing.expectApproxEqAbs(complex_output[i].re, 0.0, 1e-10);
        try std.testing.expectApproxEqAbs(complex_output[i].im, 0.0, 1e-10);
    }
}

test "RealPlan3d roundtrip" {
    const allocator = std.testing.allocator;

    var plan = try RealPlan3d.init(allocator, 8, 8, 8);
    defer plan.deinit();

    const real_size = 8 * 8 * 8;
    const complex_size = 5 * 8 * 8; // (8/2+1) * 8 * 8

    var original: [real_size]f64 = undefined;
    for (0..real_size) |i| {
        original[i] = @floatFromInt(i % 17);
    }

    var spectrum: [complex_size]Complex = undefined;
    var recovered: [real_size]f64 = undefined;

    plan.forward(&original, &spectrum);
    plan.inverse(&spectrum, &recovered);

    for (0..real_size) |i| {
        try std.testing.expectApproxEqAbs(recovered[i], original[i], 1e-9);
    }
}

test "RealPlan3d vs complex Plan3d" {
    const allocator = std.testing.allocator;

    const nx = 8;
    const ny = 8;
    const nz = 8;
    const real_size = nx * ny * nz;
    const complex_size = (nx / 2 + 1) * ny * nz;

    var rfft_plan = try RealPlan3d.init(allocator, nx, ny, nz);
    defer rfft_plan.deinit();

    var cfft_plan = try fft.Plan3d.init(allocator, nx, ny, nz);
    defer cfft_plan.deinit();

    // Test input
    var real_input: [real_size]f64 = undefined;
    for (0..real_size) |i| {
        real_input[i] = @floatFromInt((i * 7) % 23);
    }

    // RFFT
    var rfft_output: [complex_size]Complex = undefined;
    rfft_plan.forward(&real_input, &rfft_output);

    // Complex FFT (convert real to complex)
    var cfft_data: [real_size]Complex = undefined;
    for (0..real_size) |i| {
        cfft_data[i] = Complex.init(real_input[i], 0);
    }
    cfft_plan.forward(&cfft_data);

    // Compare the non-redundant part: first (nx/2+1) x-values for each (y,z)
    const nx_c = nx / 2 + 1;
    for (0..nz) |iz| {
        for (0..ny) |iy| {
            for (0..nx_c) |ix| {
                const rfft_idx = ix + nx_c * (iy + ny * iz);
                const cfft_idx = ix + nx * (iy + ny * iz);
                try std.testing.expectApproxEqAbs(rfft_output[rfft_idx].re, cfft_data[cfft_idx].re, 1e-9);
                try std.testing.expectApproxEqAbs(rfft_output[rfft_idx].im, cfft_data[cfft_idx].im, 1e-9);
            }
        }
    }
}
