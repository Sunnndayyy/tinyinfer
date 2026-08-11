# Focused M4 Pro Metal evidence

This test uses the Qwen2.5-1.5B `1536 -> 8960` projection. It compares one
decode row with 256 prefill rows. It tests PyTorch BF16 and TinyInfer Q8.

## Run the benchmark

```bash
tinyinfer benchmark --roofline
```

The command writes `.tinyinfer/roofline/results.json` and
`.tinyinfer/roofline/roofline.svg`. Open the graph with:

```bash
open .tinyinfer/roofline/roofline.svg
```

The program does these steps:

1. It makes matched Q8 and BF16 weights.
2. It checks the Q8 output against BF16.
3. It warms both operations.
4. It records interleaved samples and synchronizes MPS after each sample.
5. It writes one table, one JSON file, and one SVG graph.

The program calculates `2 * rows * input_width * output_width` FLOPs. The byte
model counts each input, weight, scale, and output one time. These are ideal
algorithmic bytes. They are not GPU counter bytes.

The synchronized M4 Pro result was:

| Rows | Path | Time | Ideal FLOP/byte | Achieved TFLOP/s | Model GB/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BF16 | 0.320 ms | 1.00 | 0.086 | 86.1 |
| 1 | Q8 | 0.275 ms | 1.88 | 0.100 | 53.3 |
| 4 | BF16 | 0.291 ms | 3.99 | 0.378 | 94.7 |
| 4 | Q8 | 0.339 ms | 7.49 | 0.324 | 43.3 |
| 16 | BF16 | 0.348 ms | 15.81 | 1.265 | 80.0 |
| 16 | Q8 | 0.736 ms | 29.44 | 0.599 | 20.3 |
| 64 | BF16 | 0.564 ms | 61.02 | 3.121 | 51.1 |
| 64 | Q8 | 2.366 ms | 110.33 | 0.744 | 6.7 |
| 256 | BF16 | 1.504 ms | 214.18 | 4.687 | 21.9 |
| 256 | Q8 | 8.950 ms | 352.38 | 0.787 | 2.2 |

At one row, one activation row uses a large weight matrix one time. The GPU
must read many weight bytes for 27.5 million FLOPs. At 256 rows, the same
weights can serve 256 activation rows. FLOPs increase almost 256 times, but the
ideal weight bytes do not. Thus, arithmetic intensity increases.

`Model GB/s` is ideal bytes divided by synchronized runtime. It is not measured
GPU traffic. The graph uses 273 GB/s advertised unified-memory bandwidth and
the best BF16 result as teaching references. Neither value is a measured
sustainable ceiling.

## Capture the four operations

```bash
tinyinfer benchmark --roofline --capture
```

The command writes these packages:

- `bf16-r1.gputrace`
- `q8-r1.gputrace`
- `bf16-r256.gputrace`
- `q8-r256.gputrace`

The files are in `.tinyinfer/roofline/captures/`. The numeric prefix can change.
Each package contains one operation. Open a package in Xcode. Use Shaders, Heat
Map, and Counters. Do not use capture replay time as benchmark time.

PyTorch implements direct capture with
[`torch.mps.profiler.metal_capture`](https://docs.pytorch.org/docs/stable/generated/torch.mps.profiler.metal_capture.html).
Xcode explains the GPU views in
[Optimizing GPU performance](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance)
and [Performance heat maps](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-performance-heatmaps-a17-m3/).

## What Xcode measured

The following values come from separate Xcode capture replay. They are not the
synchronized benchmark values.

| Path | Rows | Replay GPU time | Device read | Device write | LLC read | Main limiter |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 1 | 133.44 us | 26.26 MiB | 15.19 KiB | 26.60 MiB | Launch 33.86% |
| Q8 | 1 | 85.12 us | 13.96 MiB | 13.00 KiB | 14.22 MiB | Launch 75.60% |
| BF16 | 256 | 1.87 ms | 27.46 MiB | 2.99 MiB | 227.31 MiB | F32 math 36.50% |
| Q8 | 256 | 31.24 ms | 14.78 MiB | 6.76 MiB | 1.94 GiB | Launch 38.65% |

The Q8 prefill capture also reports Instruction at 29.15% and Integer or
Complex Math at 30.74%. No one limiter dominates that dispatch.

These values show four different effects:

- Q8 at one row reads about half as many device bytes as BF16. The LLC miss rate
  is 97.4%. Large weight traffic and launch limits both affect decode.
- BF16 at 256 rows uses a tiled matrix kernel. Its main reported limiter changes
  to F32 math. Its launch cost is no longer the main limit.
- Q8 at 256 rows reads only about 15 MiB from device memory, but it reads 1.94
  GiB from the LLC. Thus, the GPU cache reuses weights. The current row-wise
  shader still does much more on-chip work than the ideal one-read model shows.
- Q8 at 256 rows is not slow because of CPU dispatch. Xcode records one long GPU
  dispatch. Its launch, instruction, and integer-math limiter values are close.
  The next kernel test must improve row tiling and SIMD work distribution.

This evidence does not prove one universal bottleneck for all Q8 kernels. It
does prove the limit for these four operations on this M4 Pro.

## Add device traffic to the graph

Copy the Device Read and Device Write byte values from Xcode into
`.tinyinfer/roofline/counters.json`:

```json
{
  "shape": {
    "input_width": 1536,
    "output_width": 8960
  },
  "measurements": {
    "bf16-r1": {
      "device_read_bytes": 27535606,
      "device_write_bytes": 15555
    },
    "q8-r1": {
      "device_read_bytes": 14638121,
      "device_write_bytes": 13312
    }
  }
}
```

The numbers above are rounded conversions from the displayed MiB and KiB
values. For exact graph points, replace them with Xcode's byte values.
The command rejects the file if its matrix shape does not match the benchmark.

Run `tinyinfer benchmark --roofline` again. The graph automatically uses the
counter file when it exists.

The graph uses two symbols:

- A circle uses ideal algorithmic intensity.
- A square uses measured device-read plus device-write intensity.

The vertical position always uses the separate synchronized benchmark time.
This keeps capture mode out of benchmark timing.

The public Metal counter API on this M4 Pro exposes a timestamp counter set. It
does not expose the Xcode Performance Limiter and memory counter sets to this
Python test. Apple requires applications to test each device's available
[counter sets](https://developer.apple.com/documentation/metal/confirming-which-counters-and-counter-sets-a-gpu-supports).
For this small tool, Xcode is the source for these counters. The tool does not
try to parse the private `.gputrace` package format.

Remove all generated Roofline files with:

```bash
tinyinfer benchmark --roofline --clean
```

This removes only `.tinyinfer/roofline/`. The repository ignores `.tinyinfer/`.
