# TinyInfer

TinyInfer is a small, readable LLM inference server.
It aims for the space between learning basic concepts and a production engine such as vLLM, SGlang etc. 

The first model target is
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on Apple Silicon. TinyInfer downloads Qwen's tokenizer and safetensors.

## overview

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
Qwen on Apple MPS
```

## Quick start

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

Take the model for a spin:

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

TinyInfer currently has no authentication or TLS, so protect the port
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

## Benchmarking

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
inefficient: it is a baseline that a later KV cache must beat.

## Current limitations

- Qwen2 only; V0 is checked against Qwen2.5-1.5B-Instruct.
- Greedy decoding only.
- Batch size one and one in-process model.
- No KV cache; prior tokens are recomputed for every output token.
- Explicit eager attention, not SDPA or FlashAttention.
- BF16 MPS by default on Apple Silicon; CPU uses float32.
- No quantization, request queue, auth, Linux/CUDA, or distributed execution.

## Development

```bash
python -m pytest -q
ruff check .
ruff format --check .
```
