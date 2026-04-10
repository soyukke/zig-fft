# zig-fft

任意サイズ対応の純粋Zig FFTライブラリ。1Dおよび3D変換をサポート。

[English README](README.md)

## 特徴

- **任意サイズ対応** - 最適なアルゴリズムを自動選択:
  - 2の累乗: Radix-2 Cooley-Tukey (SIMD最適化)
  - Smooth数 (2^a x 3^b x 5^c): Mixed-radix Cooley-Tukey
  - その他: Bluesteinアルゴリズム
- **1D, 3D, 実数FFT** サポート
- **並列3D FFT** (スレッドプール)
- **オプションバックエンド**: FFTW3, Apple vDSP, Metal GPU

## インストール

`build.zig.zon` に追加:

```zig
.dependencies = .{
    .fft = .{
        .url = "https://github.com/soyukke/zig-fft/archive/main.tar.gz",
        .hash = "...",  // `zig fetch` で取得
    },
},
```

`build.zig` で:

```zig
const fft_dep = b.dependency("fft", .{
    .target = target,
    .optimize = optimize,
});
your_module.addImport("fft", fft_dep.module("fft"));
```

## 使い方

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

## オプションバックエンド

### FFTW3

```bash
zig build -Dfftw-include=/path/to/fftw/include -Dfftw-lib=/path/to/fftw/lib
```

### vDSP (macOS)

macOSではAccelerateフレームワーク経由で自動的に利用可能。

### Metal GPU (macOS)

Metal対応ハードウェアを搭載したmacOSで利用可能。Metal bridge (Objective-Cソース) が必要。

## ビルド

```bash
zig build          # ライブラリのビルド
zig build test     # テスト実行
```

## ライセンス

MIT
