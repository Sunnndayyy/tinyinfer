# Q8 Metal benchmark on M4 Pro

These results compare TinyInfer's fused Q8 Metal operation with PyTorch's BF16
`F.linear` on the same Apple M4 Pro. The operator measurements were taken after
warmup. The executable warms both paths together and reports the median from
interleaved, individually synchronized samples. Reproduce the fixed Qwen decode
shapes from the repository root:

```bash
.venv/bin/python benchmarks/q8_mps_linear.py
```

| Activation x weight shape | Q8 | BF16 | BF16 / Q8 |
| --- | ---: | ---: | ---: |
| `[1, 1536] x [8960, 1536]` | 0.231 ms | 0.238 ms | 1.03x |
| `[1, 1536] x [151936, 1536]` | 1.196 ms | 2.072 ms | 1.732x |

The separate real-model reverse-order decode measurement for
Qwen2.5-1.5B-Instruct reached 42.697979 tok/s with BF16 and 68.574666 tok/s with
Q8 after the packed embedding landed, a roughly 1.606x decode speedup.

Prefill did not improve. Time to first token was 0.04024 seconds for BF16 and
0.09874 seconds for Q8, so the Q8 TTFT was about 2.45x slower. Keep that
regression separate from steady-state decode throughput: a decode win does not
make the prefill path faster.

These measurements are same-machine evidence, not cross-machine promises. The
script prints the active PyTorch version and accepts `--warmup`, `--repetitions`,
and `--seed` so later runs can state their measurement settings.
