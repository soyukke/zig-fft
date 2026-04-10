//! FFTW3 FFT wrapper for Zig
//!
//! Provides a 3D FFT implementation using the FFTW3 library.
//! FFTW3 is highly optimized for arbitrary sizes including mixed-radix.
//!
//! This module requires FFTW3 to be linked. Build with:
//!   zig build -Dfftw-include=/path/to/fftw/include -Dfftw-lib=/path/to/fftw/lib

const std = @import("std");
pub const Complex = @import("complex.zig").Complex;

// Build option to enable FFTW (set by build.zig when FFTW paths are provided)
const enable_fftw = @import("fftw_options").enable_fftw;

// FFTW3 C bindings (only imported when FFTW is enabled)
const c = if (enable_fftw) @cImport({
    @cInclude("fftw3.h");
}) else struct {
    // Stub types for when FFTW is not available
    const fftw_complex = [2]f64;
    const fftw_plan = ?*anyopaque;
    const FFTW_FORWARD: c_int = -1;
    const FFTW_BACKWARD: c_int = 1;
    const FFTW_ESTIMATE: c_uint = 64;

    fn fftw_plan_dft_3d(_: c_int, _: c_int, _: c_int, _: [*c]fftw_complex, _: [*c]fftw_complex, _: c_int, _: c_uint) fftw_plan {
        return null;
    }
    fn fftw_execute_dft(_: fftw_plan, _: [*c]fftw_complex, _: [*c]fftw_complex) void {}
    fn fftw_destroy_plan(_: fftw_plan) void {}
};

/// FFTW complex type (interleaved double[2])
const FftwComplex = c.fftw_complex;

/// Global mutex for FFTW planner thread safety.
/// FFTW's planner is NOT thread-safe by default, so we must serialize
/// all plan creation/destruction calls.
var planner_mutex: std.Thread.Mutex = .{};

/// FFTW 3D Plan
pub const FftwPlan3d = struct {
    nx: usize,
    ny: usize,
    nz: usize,
    forward_plan: c.fftw_plan,
    inverse_plan: c.fftw_plan,
    buffer: []Complex,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !FftwPlan3d {
        if (!enable_fftw) {
            return error.FftwNotAvailable;
        }

        if (nx == 0 or ny == 0 or nz == 0) return error.InvalidSize;

        const size = nx * ny * nz;
        const buffer = try allocator.alloc(Complex, size);
        errdefer allocator.free(buffer);

        // Cast buffer to FFTW complex pointer
        const fftw_ptr: [*c]FftwComplex = @ptrCast(buffer.ptr);

        // Lock mutex for thread-safe plan creation
        planner_mutex.lock();
        defer planner_mutex.unlock();

        // Create FFTW plans (FFTW_ESTIMATE for fast planning)
        const forward_plan = c.fftw_plan_dft_3d(
            @intCast(nz),
            @intCast(ny),
            @intCast(nx),
            fftw_ptr,
            fftw_ptr,
            c.FFTW_FORWARD,
            c.FFTW_MEASURE,
        );
        if (forward_plan == null) {
            return error.FftwPlanFailed;
        }
        errdefer c.fftw_destroy_plan(forward_plan);

        const inverse_plan = c.fftw_plan_dft_3d(
            @intCast(nz),
            @intCast(ny),
            @intCast(nx),
            fftw_ptr,
            fftw_ptr,
            c.FFTW_BACKWARD,
            c.FFTW_MEASURE,
        );
        if (inverse_plan == null) {
            return error.FftwPlanFailed;
        }

        return .{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .forward_plan = forward_plan,
            .inverse_plan = inverse_plan,
            .buffer = buffer,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FftwPlan3d) void {
        if (enable_fftw) {
            // Lock mutex for thread-safe plan destruction
            planner_mutex.lock();
            defer planner_mutex.unlock();
            c.fftw_destroy_plan(self.forward_plan);
            c.fftw_destroy_plan(self.inverse_plan);
        }
        self.allocator.free(self.buffer);
    }

    pub fn forward(self: *FftwPlan3d, data: []Complex) void {
        if (!enable_fftw) return;

        const size = self.nx * self.ny * self.nz;
        if (data.len != size) return;

        const fftw_ptr: [*c]FftwComplex = @ptrCast(data.ptr);
        c.fftw_execute_dft(self.forward_plan, fftw_ptr, fftw_ptr);
    }

    pub fn inverse(self: *FftwPlan3d, data: []Complex) void {
        if (!enable_fftw) return;

        const size = self.nx * self.ny * self.nz;
        if (data.len != size) return;

        const fftw_ptr: [*c]FftwComplex = @ptrCast(data.ptr);
        c.fftw_execute_dft(self.inverse_plan, fftw_ptr, fftw_ptr);

        // Normalize (FFTW doesn't normalize inverse)
        const scale = 1.0 / @as(f64, @floatFromInt(size));
        for (data) |*v| {
            v.re *= scale;
            v.im *= scale;
        }
    }

    /// Check if FFTW backend is available
    pub fn isAvailable() bool {
        return enable_fftw;
    }
};

// ============== Tests ==============

test "FftwPlan3d roundtrip" {
    if (!enable_fftw) return; // Skip if FFTW not available

    const allocator = std.testing.allocator;

    var plan = try FftwPlan3d.init(allocator, 8, 8, 8);
    defer plan.deinit();

    var data: [512]Complex = undefined;
    var original: [512]Complex = undefined;
    for (0..512) |i| {
        data[i] = Complex.init(@floatFromInt(i), 0);
        original[i] = data[i];
    }

    plan.forward(&data);
    plan.inverse(&data);

    for (0..512) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}

test "FftwPlan3d 24x24x24" {
    if (!enable_fftw) return; // Skip if FFTW not available

    const allocator = std.testing.allocator;

    var plan = try FftwPlan3d.init(allocator, 24, 24, 24);
    defer plan.deinit();

    const size = 24 * 24 * 24;
    var data = try allocator.alloc(Complex, size);
    defer allocator.free(data);
    var original = try allocator.alloc(Complex, size);
    defer allocator.free(original);

    for (0..size) |i| {
        data[i] = Complex.init(@floatFromInt(i % 17), @floatFromInt(i % 13));
        original[i] = data[i];
    }

    plan.forward(data);
    plan.inverse(data);

    for (0..size) |i| {
        try std.testing.expectApproxEqAbs(data[i].re, original[i].re, 1e-9);
        try std.testing.expectApproxEqAbs(data[i].im, original[i].im, 1e-9);
    }
}
