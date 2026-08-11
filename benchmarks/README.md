# Q8 Metal benchmark on M4 Pro

These measurements compare TinyInfer's fused Q8 Metal path with PyTorch BF16
on the same Apple M4 Pro using PyTorch 2.13.0. Both sweeps warm the paths,
synchronize every timed MPS sample, report medians, and alternate which path
runs first.

## Operator row sweep

Run every distinct Qwen2.5-1.5B linear shape:

```bash
.venv/bin/python benchmarks/q8_mps_linear.py \
  --warmup 10 --repetitions 20 --seed 0
```

The tied vocabulary projection is intentionally measured only at one row.
TinyInfer's `next_token_logits` projects the final hidden state, not every
prompt row.

| Operation | Rows | Input | Output | Q8 | BF16 | BF16 / Q8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention k/v | 1 | 1536 | 256 | 0.162 ms | 0.148 ms | 0.918x |
| attention k/v | 8 | 1536 | 256 | 0.190 ms | 0.171 ms | 0.903x |
| attention k/v | 32 | 1536 | 256 | 0.298 ms | 0.250 ms | 0.840x |
| attention k/v | 128 | 1536 | 256 | 0.778 ms | 0.320 ms | 0.411x |
| attention k/v | 512 | 1536 | 256 | 0.992 ms | 0.314 ms | 0.317x |
| attention q/o | 1 | 1536 | 1536 | 0.167 ms | 0.170 ms | 1.018x |
| attention q/o | 8 | 1536 | 1536 | 0.222 ms | 0.182 ms | 0.818x |
| attention q/o | 32 | 1536 | 1536 | 0.453 ms | 0.216 ms | 0.477x |
| attention q/o | 128 | 1536 | 1536 | 1.106 ms | 0.307 ms | 0.278x |
| attention q/o | 512 | 1536 | 1536 | 3.118 ms | 0.629 ms | 0.202x |
| MLP gate/up | 1 | 1536 | 8960 | 0.182 ms | 0.202 ms | 1.113x |
| MLP gate/up | 8 | 1536 | 8960 | 0.398 ms | 0.250 ms | 0.627x |
| MLP gate/up | 32 | 1536 | 8960 | 1.232 ms | 0.365 ms | 0.296x |
| MLP gate/up | 128 | 1536 | 8960 | 4.466 ms | 0.849 ms | 0.190x |
| MLP gate/up | 512 | 1536 | 8960 | 17.262 ms | 2.655 ms | 0.154x |
| MLP down | 1 | 8960 | 1536 | 0.187 ms | 0.235 ms | 1.254x |
| MLP down | 8 | 8960 | 1536 | 0.399 ms | 0.254 ms | 0.637x |
| MLP down | 32 | 8960 | 1536 | 1.188 ms | 0.425 ms | 0.358x |
| MLP down | 128 | 8960 | 1536 | 4.224 ms | 0.945 ms | 0.224x |
| MLP down | 512 | 8960 | 1536 | 16.840 ms | 2.768 ms | 0.164x |
| tied vocabulary | 1 | 1536 | 151936 | 1.154 ms | 1.977 ms | 1.713x |

The large MLP projections win with Q8 at one row and lose by eight rows. The
attention projections are already roughly tied or slower at one row. This is
operator evidence for a decode-oriented crossover between one and eight rows.
It does not identify the exact hardware limit in the kernel.

## Focused Roofline and Metal capture

Use the focused experiment to generate JSON, an SVG Roofline plot, and four
direct Xcode GPU captures for the 1536 to 8960 projection:

```bash
.venv/bin/python benchmarks/q8_mps_roofline.py \
  --json /tmp/tinyinfer-roofline.json \
  --plot /tmp/tinyinfer-roofline.svg

MTL_CAPTURE_ENABLED=1 .venv/bin/python benchmarks/q8_mps_roofline.py \
  --metal-capture-dir /tmp/tinyinfer-metal-captures
```

See [the focused Roofline evidence](q8_mps_roofline.md) for the counter input,
measured results, and proof limits. Capture replay time is not benchmark time.

## Exact-token real-model sweep

Run matched BF16 and Q8 checkpoints with SDPA and a contiguous cache:

```bash
.venv/bin/python benchmarks/q8_end_to_end.py \
  Qwen/Qwen2.5-1.5B-Instruct \
  /path/to/qwen2.5-1.5b-q8 \
  --lengths 1 8 32 128 512 \
  --max-new-tokens 32 --warmup 1 --repetitions 3
```

The script constructs exact token-ID sequences, so its TTFT starts from token
IDs and excludes tokenizer time. `First forward` isolates the synchronized
prompt forward with a fresh cache. TTFT also includes generation setup, argmax,
and decoding the first token. A run fails if either format stops before the
requested output length because unequal decode spans are not comparable.

| Input tokens | BF16 first forward | Q8 first forward | BF16 TTFT | Q8 TTFT | BF16 decode | Q8 decode | Q8 / BF16 decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 22.58 ms | 15.12 ms | 23.05 ms | 15.19 ms | 44.70 tok/s | 69.17 tok/s | 1.547x |
| 8 | 25.20 ms | 35.74 ms | 26.33 ms | 35.90 ms | 45.25 tok/s | 69.36 tok/s | 1.533x |
| 32 | 40.75 ms | 111.46 ms | 40.47 ms | 112.34 ms | 45.06 tok/s | 68.70 tok/s | 1.524x |
| 128 | 94.74 ms | 419.29 ms | 84.59 ms | 418.67 ms | 43.14 tok/s | 63.48 tok/s | 1.471x |
| 512 | 302.40 ms | 1688.00 ms | 289.87 ms | 1688.81 ms | 37.60 tok/s | 52.56 tok/s | 1.398x |

The matched end-to-end crossover is also between one and eight input tokens.
The existing `decode` leaderboard prompt is exactly 28 chat-formatted tokens,
so the 32-token result closely reproduces its TTFT split: about 40 ms for BF16
and 112 ms for Q8 here, versus 40 ms and 103 ms in `BENCHMARKS.md`.
First-forward time accounts for almost all of the Q8 gap. This places the
regression in prompt model execution, not tokenization or first-token text
handling.

Q8 remains faster for steady one-token decode at every measured prompt length,
although the advantage narrows as the cached context grows. These results prove
the crossover only for this model, implementation, software version, and M4 Pro.
They do not prove whether dispatch, occupancy, memory access, or arithmetic is
the limiting mechanism inside the Metal kernel.
