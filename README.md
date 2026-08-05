# TinyInfer

TinyInfer is a small, readable LLM inference server.
It aims for the space between learning basic concepts and a production engine such as vLLM, SGlang etc. 

The first model target is
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on Apple Silicon. TinyInfer downloads Qwen's tokenizer and safetensors.

## overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for the optimization routes and the order
in which their reference and experimental implementations should be added.

```text
remote client
    |
    | POST /v1/chat/completions
    v
server.py       HTTP JSON in, streamed SSE out
    |
    v
engine.py       validate request -> select decoder
    |
    v
decoding/       repeat: model -> choose token -> append -> emit
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
  --decoding autoregressive \
  --kv-cache contiguous \
  --max-new-tokens 24
```

Start the server:

```bash
tinyinfer serve Qwen/Qwen2.5-1.5B-Instruct \
  --decoding autoregressive \
  --kv-cache contiguous \
  --host 127.0.0.1 \
  --port 8000
```

Then open the interactive HTTP client from a second terminal:

```bash
tinyinfer chat \
  --host http://127.0.0.1:8000
```

The client shows the loaded model and runtime configuration, streams each reply,
and keeps the full conversation in every later request. Use `/clear` to reset
the conversation or `/quit` to leave. Override the system message and output
limit with `--system` and `--max-tokens`.

The displayed TTFT is measured by the client from request dispatch to the first
streamed token, so it includes network, admission, request parsing, tokenization,
and prefill. The terminal SSE metadata reports `server_ttft_seconds` separately
for server-side generation work. It also emits the deprecated
`time_to_first_token_seconds` alias with the same server-only value for older
clients; `ChatCompletion.time_to_first_token` remains the matching deprecated
client accessor.

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
  --kv-cache contiguous \
  --max-new-tokens 16 \
  --warmup 1 \
  --repetitions 3
```

Select decoding with `--decoding autoregressive`, and compare cache implementations
with `--kv-cache none`, `contiguous`, or `paged`. Add `--json` for machine-readable
results. Reports contain the selected decoding and cache routes, cache allocation,
TTFT, inter-token latency, end-to-end latency, output tokens per second, percentile
summaries, and the model/device/dtype/software metadata needed to interpret them.

`none` recomputes the whole sequence for every new token and is the correctness
baseline. `contiguous` preallocates a request-local cache and is the default.
`paged` allocates fixed-size blocks lazily, but currently gathers them before
eager attention; it teaches paging and is not yet a PagedAttention speedup.

## Current limitations

- Qwen2 only; V0 is checked against Qwen2.5-1.5B-Instruct.
- Greedy decoding only.
- Batch size one and one in-process model.
- Request-local contiguous and educational paged KV caches; no shared cache pool.
- Explicit eager attention, not SDPA or FlashAttention.
- BF16 MPS by default on Apple Silicon; CPU uses float32.
- No quantization, request queue, auth, Linux/CUDA, or distributed execution.

## Development

```bash
python -m pytest -q
ruff check .
ruff format --check .
```
