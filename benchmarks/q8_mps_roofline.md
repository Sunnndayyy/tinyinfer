# Focused M4 Pro Metal evidence

This capture pass profiles the same Qwen2.5-1.5B `1536 -> 8960` projection as
the synchronized [Roofline benchmark](README.md). Capture timing is diagnostic
only and is never used as benchmark timing.

## Commands

Run Metal System Trace for Q8 and BF16 at rows 1 and 256:

```bash
xcrun xctrace record --template 'Metal System Trace' \
  --output q8-r1.trace --launch -- \
  .venv/bin/python benchmarks/q8_mps_linear.py \
  --rows 1 --profile-path q8 --profile-iterations 20

xcrun xctrace record --template 'Metal System Trace' \
  --output bf16-r256.trace --launch -- \
  .venv/bin/python benchmarks/q8_mps_linear.py \
  --rows 256 --profile-path bf16 --profile-iterations 20
```

Repeat the command for the other path/row combinations. Use the Logging
template to export PyTorch's BF16 MPS intervals:

```bash
xcrun xctrace record --template Logging \
  --output bf16-r1-logging.trace --launch -- \
  .venv/bin/python benchmarks/q8_mps_linear.py \
  --rows 1 --profile-path bf16 --profile-iterations 20

xcrun xctrace export --input bf16-r1-logging.trace --toc
xcrun xctrace export --input bf16-r1-logging.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost-interval"]'
```

PyTorch's
[`torch.mps.profiler.profile`](https://docs.pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html)
emits interval and event signposts with `wait_until_completed=True`. PyTorch
warns that this option hurts performance, so the synchronized benchmark runs
separately. Apple's
[signpost documentation](https://developer.apple.com/documentation/os/recording-performance-data)
explains the interval model.

## Results

| Path | Rows | Synchronized time | MPS interval | GPU execution | CPU to GPU | CPU encode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 0.256 ms | 0.512 ms median | 0.158 ms median | 0.079 ms median | 0.013 ms median |
| BF16 | 256 | 1.492 ms | 1.357 ms median | 1.243 ms median | 0.071 ms median | 0.019 ms median |
| Q8 | 1 | 0.205 ms | unavailable | 0.162 ms/dispatch mean | 0.335 ms/batch | 0.120 ms/batch |
| Q8 | 256 | 9.086 ms | unavailable | 8.529 ms/dispatch mean | 0.115 ms/batch | 0.116 ms/batch |

The synchronized column is the normal 100-repetition median. All other columns
come from separate 20-operation capture runs. BF16 reports per-operation
medians. The custom Q8 path did not emit a `PyTorchMPS` operation signpost;
Metal recorded one command buffer for 20 dispatches. Q8 GPU execution is the
command-buffer total divided by 20, while its latency and encode values are per
batch.

## Measured evidence versus inference

- At one BF16 row, 0.079 ms CPU-to-GPU latency is material beside 0.158 ms GPU
  execution. Dispatch and scheduling overhead matter at decode scale. The low
  algorithmic intensity and Q8's smaller synchronized time are consistent with
  memory pressure, but memory bandwidth is not counter-proven.
- At 256 BF16 rows, GPU execution grows to 1.243 ms while CPU-to-GPU latency
  stays near 0.071 ms. Launch overhead is no longer primary, and synchronized
  throughput rises from 0.107 to 4.722 TFLOP/s. Increasingly compute-bound is a
  Roofline inference from rising arithmetic intensity, not an ALU counter.
- At 256 Q8 rows, GPU work takes 170.582 ms for 20 dispatches while
  command-buffer startup takes 0.115 ms. Dispatch overhead is not the main Q8
  prefill problem. The shader maps 2,240 threadgroups per row without cross-row
  tiling. It does not explicitly realize the ideal model's weight reuse.
- Metal System Trace exported `0 Bytes` for these private-buffer compute
  intervals. Adding `Metal GPU Counters` selected Performance Limiters but
  reported `Selected counter profile is not supported on target device`; its
  counter tables were empty. Occupancy, memory traffic/bandwidth, and ALU
  efficiency are unavailable, not zero. Apple's
  [GPU counter documentation](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
  requires checking which counter sets each device supports.

The trace proves that Q8 prefill time is spent executing the GPU kernel rather
than launching it. It does not distinguish repeated memory traffic, occupancy,
or arithmetic efficiency as the in-kernel limiter.
