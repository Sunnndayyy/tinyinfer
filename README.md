# TinyInfer

TinyInfer is a small, readable LLM inference server built around a real model.
It aims for the space between micrograd and a production engine such as vLLM:
useful enough to run, small enough to read and change.

The first target is
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on Apple Silicon. TinyInfer downloads Qwen's tokenizer and safetensor files but
does **not** use Transformers, vLLM, SGLang, or llama.cpp to execute the model.

## The whole system

```text
remote client
    |
    | POST /v1/chat/completions
    v
server.py       HTTP JSON in, streamed SSE out
    |
    v
engine.py       repeat: model -> choose token -> append -> emit
    |
    v
model.py        tokens -> Qwen decoder layers -> next-token logits
    |
    v
real Qwen weights on Apple MPS
```

That is the mental model. `artifacts.py` finds files; `tokenizer.py` translates
between text and token IDs. They are helpers, not extra runtime layers.

## Quick start

TinyInfer requires Python 3.11 or newer. This version has been exercised on an
Apple M4 Pro with Python 3.14 and PyTorch's MPS backend.

```bash
git clone <this-repository-url> tinyinfer
cd tinyinfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Download the default model (roughly 3 GB):

```bash
tinyinfer download Qwen/Qwen2.5-1.5B-Instruct
```

Prove local generation before adding networking:

```bash
tinyinfer generate Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "What does a KV cache save?" \
  --max-new-tokens 24
```

Start the server:

```bash
tinyinfer serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000
```

Then use the HTTP client from a second terminal:

```bash
tinyinfer chat \
  --host http://127.0.0.1:8000 \
  --prompt "Explain prefill and decode in plain language."
```

For LAN access, bind to `0.0.0.0 --allow-remote` and use the Mac's LAN address
in the client. TinyInfer V0 has no authentication or TLS, so protect the port
with a firewall and do not expose it directly to the internet.

## OpenAI-shaped API

V0 implements a deliberately narrow, text-only subset of chat completions:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 16,
    "stream": true
  }'
```

The route and SSE chunks are OpenAI-shaped; V0 does not claim complete API
compatibility. One request owns the model at a time and excess work receives
HTTP 503 rather than running concurrent MPS forwards.

## Benchmark the baseline

```bash
tinyinfer bench Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "What does a KV cache save?" \
  --max-new-tokens 16 \
  --warmup 1 \
  --repetitions 3
```

Add `--json` for machine-readable results. Reports contain TTFT, inter-token
latency, end-to-end latency, output tokens per second, percentile summaries,
and the model/device/dtype/software metadata needed to interpret them.

V0 recomputes the whole sequence for every new token. This is intentionally
inefficient: it is the honest baseline that a later KV cache must beat.

## Read the code in this order

1. [`tinyinfer/tokenizer.py`](tinyinfer/tokenizer.py) — exact ChatML text and
   text-to-token conversion.
2. [`tinyinfer/model.py`](tinyinfer/model.py) — embeddings, RMSNorm, RoPE,
   grouped-query causal attention, the gated MLP, residuals, and tied logits.
3. [`tinyinfer/engine.py`](tinyinfer/engine.py) — the autoregressive generation
   loop and the difference between EOS and token-limit termination.
4. [`tinyinfer/server.py`](tinyinfer/server.py) — request validation, one-model
   admission, non-streaming JSON, and streaming SSE.
5. [`tinyinfer/benchmark.py`](tinyinfer/benchmark.py) — where each latency
   measurement starts and stops.

The separation is intentional: the model knows tensor math but not HTTP; the
engine knows generation but not sockets; the server knows the API but not how
attention works. Future optimization experiments therefore have a clear home.

## Current limitations

- Qwen2 only; V0 is checked against Qwen2.5-1.5B-Instruct.
- Greedy decoding only.
- Batch size one and one in-process model.
- No KV cache; prior tokens are recomputed for every output token.
- Explicit eager attention, not SDPA or FlashAttention.
- BF16 MPS by default on Apple Silicon; CPU uses float32.
- No quantization, request queue, auth, Linux/CUDA, or distributed execution.

These limitations are the optimization curriculum. Each should arrive as a
small, readable change with before-and-after measurements.

## Development

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

See the [real-model V0 plan](docs/plans/2026-08-05-001-feat-tinyinfer-real-model-plan.md).
