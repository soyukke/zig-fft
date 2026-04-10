const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const target_os = target.result.os.tag;

    // FFTW3 options
    const fftw_include = b.option([]const u8, "fftw-include", "Path to FFTW3 include directory");
    const fftw_lib = b.option([]const u8, "fftw-lib", "Path to FFTW3 library directory");
    const enable_fftw = fftw_include != null and fftw_lib != null;

    const fftw_options = b.addOptions();
    fftw_options.addOption(bool, "enable_fftw", enable_fftw);

    // Main module
    const mod = b.addModule("fft", .{
        .root_source_file = b.path("src/fft.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "fftw_options", .module = fftw_options.createModule() },
        },
    });

    // Link FFTW if available
    if (fftw_include) |inc| mod.addIncludePath(.{ .cwd_relative = inc });
    if (fftw_lib) |lib| mod.addLibraryPath(.{ .cwd_relative = lib });
    if (enable_fftw) mod.linkSystemLibrary("fftw3", .{});

    // Link system frameworks on macOS (vDSP / Accelerate)
    if (target_os == .macos) {
        mod.linkFramework("Accelerate", .{});
    }

    // Tests
    const tests = b.addTest(.{
        .root_module = mod,
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
