# TinyInfer architecture

TinyInfer keeps one visible path from an HTTP request to generated tokens. Each
optimization changes one named part of that path and remains comparable with a
slow reference implementation.

```text
server configuration
        |
        v
      engine
        |
        v
    scheduler  -------- chooses which request runs
        |
        v
     decoder   -------- decides how tokens are proposed and verified
        |
        v
      model
      /   \
     v     v
attention  KV cache --- calculates attention and stores past K/V tensors
        |
        v
     sampler   -------- chooses a token from logits
        |
        v
  streamed token
```

## Code map

| Route | Reference implementation | Later experiments |
| --- | --- | --- |
| `kv_cache/` | `none.py`, then `contiguous.py` | `paged.py` |
| `attention/` | `eager.py` | `sdpa.py`, `flash.py` |
| `decoding/` | `autoregressive.py` | `speculative.py`, `dspark/` |
| `scheduling/` | `serial.py` | `continuous.py` |
| `sampling/` | `greedy.py` | `top_k.py`, `top_p.py` |
| `quantization/` | full-precision model | `int8.py`, `int4.py` |
| `parallelism/` | one device | `tensor.py`, `pipeline.py` |

`runtime.py` will be the single place that turns server flags into these
implementation choices and rejects incompatible combinations. Keep selection
explicit; do not add dynamic plugin discovery or a universal `Technique` base
class.

## Implementation order

1. Keep the full-prefix generation path as `kv_cache/none.py`. **Implemented.**
2. Add a per-request contiguous cache in `kv_cache/contiguous.py`. **Implemented.**
3. Prove uncached, contiguous, and paged paths produce matching greedy logits,
   then benchmark them. **Implemented.**
4. Move eager attention behind `attention/eager.py` and add SDPA as the second
   implementation.
5. Move the current generation loop behind `decoding/autoregressive.py`.
6. Add sampling alternatives, then continuous request scheduling.
7. Add ordinary speculative decoding before attempting DSpark.
8. Replace the educational paged allocator with shared block pools, sequence
   block tables, and page-aware attention after continuous scheduling exists;
   add quantization and multi-device execution when their hardware foundations
   exist.

## Rules for implementing papers

- One atomic algorithm can be one file. A paper that changes several parts of
  inference gets a small package, as DSpark does.
- Every optimized path must match a readable reference path before its speed is
  considered.
- Model- or training-dependent techniques must declare their checkpoint
  requirements; they are not ordinary runtime flags.
- Benchmarks must record every selected implementation so results remain
  comparable.
- A scaffolded module is a destination, not a claim that the technique works.

## Current boundary

KV-cache routing is live in `model.py`, `engine.py`, and the three `kv_cache/`
implementations. The other packages remain roadmap markers. Move more behavior
behind a route only when that route has correctness tests and a useful
comparison implementation.
