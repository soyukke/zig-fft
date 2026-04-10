# zig-fft

Pure Zig FFT library supporting arbitrary sizes with 1D and 3D transforms.

[日本語版 README](README.ja.md)

## Features

- **Arbitrary size support** - automatically selects the best algorithm:
  - Power of 2: Radix-2 Cooley-Tukey (SIMD optimized)
  - Smooth numbers (2^a x 3^b x 5^c): Mixed-radix Cooley-Tukey
  - Other sizes: Bluestein's algorithm
- **1D, 3D, and real FFT** support
- **Parallel 3D FFT** with thread pool
- **Optional backends**: FFTW3, Apple vDSP, Metal GPU

## Installation

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .fft = .{
        .url = "https://github.com/soyukke/zig-fft/archive/main.tar.gz",
        .hash = "...",  // run `zig fetch` to get this
    },
},
```

Then in `build.zig`:

```zig
const fft_dep = b.dependency("fft", .{
    .target = target,
    .optimize = optimize,
});
your_module.addImport("fft", fft_dep.module("fft"));
```

## Usage

### 1D FFT

```zig
const fft = @import("fft");

var plan = try fft.Plan1d.init(allocator, 1024);
defer plan.deinit();

plan.forward(&data);
plan.inverse(&data);
```

### 3D FFT

```zig
var plan = try fft.Plan3d.init(allocator, 32, 32, 32);
defer plan.deinit();

plan.forward(&data);
plan.inverse(&data);
```

## Optional Backends

### FFTW3

```bash
zig build -Dfftw-include=/path/to/fftw/include -Dfftw-lib=/path/to/fftw/lib
```

### vDSP (macOS)

Automatically available on macOS via the Accelerate framework.

### Metal GPU (macOS)

Available on macOS with Metal-capable hardware. Requires the Metal bridge (Objective-C source).

## Building

```bash
zig build          # build library
zig build test     # run tests
```

## License

MIT
