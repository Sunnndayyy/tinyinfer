# Decode-to-prefill Roofline on M4 Pro

These results compare TinyInfer's fused Q8 Metal operation with PyTorch's BF16
`F.linear` on the same Qwen2.5-1.5B MLP projection. The executable warms both
paths together and reports the median from interleaved, individually
synchronized samples:

```bash
.venv/bin/python benchmarks/q8_mps_linear.py \
  --json /tmp/tinyinfer-roofline.json \
  --plot /tmp/tinyinfer-roofline.svg
```

The command also checks the Q8 result against BF16 before it starts timing.
Open the SVG in a browser. The circle position uses ideal algorithmic bytes.

Shape: `[rows, 1536] x [8960, 1536]`. PyTorch 2.13.0, 10 warmups, 100
repetitions, seed 0.

| Rows | Path | Time | Ideal FLOP/byte | Achieved TFLOP/s | Model GB/s | Model side |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | BF16 | 0.256 ms | 1.00 | 0.107 | 107.5 | memory |
| 1 | Q8 | 0.205 ms | 1.88 | 0.134 | 71.4 | memory |
| 4 | BF16 | 0.288 ms | 3.99 | 0.383 | 95.9 | memory |
| 4 | Q8 | 0.402 ms | 7.49 | 0.274 | 36.6 | memory |
| 16 | BF16 | 0.327 ms | 15.81 | 1.345 | 85.1 | memory |
| 16 | Q8 | 0.695 ms | 29.44 | 0.634 | 21.5 | compute |
| 64 | BF16 | 0.525 ms | 61.02 | 3.352 | 54.9 | compute |
| 64 | Q8 | 2.335 ms | 110.33 | 0.754 | 6.8 | compute |
| 256 | BF16 | 1.492 ms | 214.18 | 4.722 | 22.0 | compute |
| 256 | Q8 | 9.086 ms | 352.38 | 0.776 | 2.2 | compute |

One decode row performs about 27.5 million matmul FLOPs while reading a large
weight matrix for that single row. With more prefill rows, the same weights can
serve many activation rows, so ideal arithmetic intensity rises from 1.00 to
214.18 FLOP/byte for BF16. The optimized BF16 operation turns that opportunity
into 4.722 TFLOP/s at 256 rows.

The byte model counts each input, weight, scale, and output exactly once. Q8
counts one INT8 byte per weight and one FP16 scale per group of 32 weights.
`model GB/s` divides this ideal byte count by synchronized runtime; it is not a
Metal hardware counter and does not reveal cache traffic or repeated loads.

The teaching crossover uses Apple's advertised
[273 GB/s unified-memory bandwidth](https://support.apple.com/en-ie/121553), not
measured sustainable bandwidth, and the best BF16 result in this sweep as a
4.722 TFLOP/s lower-bound compute reference, not a peak ceiling. This puts the
estimated crossover at 17.30 FLOP/byte. `Model side` is a Roofline classification
under those assumptions, not counter evidence.

TinyInfer's current Q8 shader maps rows independently and does not explicitly
tile rows to reuse weights. Its ideal arithmetic intensity therefore describes
the prefill opportunity, not traffic the kernel is proven to achieve. The Q8
path wins at one row here, then falls behind BF16 as rows increase.

## Focused Metal trace

Capture mode is separate from benchmark timing. This command writes four
direct Xcode GPU captures. Each capture has one test operation:

```bash
MTL_CAPTURE_ENABLED=1 .venv/bin/python benchmarks/q8_mps_linear.py \
  --metal-capture-dir /tmp/tinyinfer-metal-captures
```

Open each `.gputrace` package in Xcode. Use the Counters view to inspect device
traffic and performance limiters. The benchmark can add copied device traffic
to its graph. See the [focused profiling evidence](q8_mps_roofline.md) for the
counter file, commands, measurements, and proof boundaries. Capture replay time
is not used as benchmark time.

The separate real-model reverse-order decode measurement for
Qwen2.5-1.5B-Instruct reached 42.697979 tok/s with BF16 and 68.574666 tok/s with
Q8 after the packed embedding landed, a roughly 1.606x decode speedup.

Prefill did not improve in the separate real-model measurement. Time to first
token was 0.04024 seconds for BF16 and 0.09874 seconds for Q8, so the Q8 TTFT was
about 2.45x slower. Keep that regression separate from steady-state decode
throughput: a decode win does not make the prefill path faster.

These measurements are same-machine evidence, not cross-machine promises. The
script prints the active PyTorch version and accepts `--warmup`, `--repetitions`,
and `--seed` so later runs can state their measurement settings.
