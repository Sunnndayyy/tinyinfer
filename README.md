<div align="center">

<img alt="TinyInfer" src="/logo_tinyinfer.svg" width="50%" height="50%">

</div>

A small, readable LLM inference engine and server built from first principles
It aims to be somewhere in the middle of learning basic concepts and a production engine such as vLLM, SGlang etc. 

The first model target is [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on Apple Silicon. 

Why? It has a simple model archtecture and is good place to start, the goal is to add more model variety with increasing complexity throughout the course of the project. 

---

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

Convert it to TinyInfer's Q8 Safetensors format:

```bash
tinyinfer quantize Qwen/Qwen2.5-1.5B-Instruct \
  --format q8 \
  --output .tinyinfer/models/qwen2.5-1.5b-q8
```

Take the model for a spin:

```bash
tinyinfer generate .tinyinfer/models/qwen2.5-1.5b-q8 \
  --device mps \
  --dtype bfloat16 \
  --prompt "What does weight quantization save?" \
  --max-new-tokens 8
```

Start the server:

```bash
tinyinfer serve Qwen/Qwen2.5-1.5B-Instruct \
  --decoding autoregressive \
  --attention sdpa \
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
for server-side generation work.

TinyInfer currently has no authentication or TLS

---

## OpenAI compatible API

TinyInfer implements a narrow, text-only subset of chat completions:

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

Currenty one request owns the model at a time and excess work receives
HTTP 503 rather than running concurrent MPS forwards.

---

## Benchmarking

```bash
tinyinfer bench Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "What does a KV cache save?" \
  --attention sdpa \
  --kv-cache contiguous \
  --max-new-tokens 16 \
  --warmup 1 \
  --repetitions 3
```

Select decoding with `--decoding autoregressive`, compare attention with
`--attention eager` or `sdpa`, and compare cache implementations with
`--kv-cache none`, `contiguous`, or `paged`. Add `--json` for machine-readable
results. Reports contain the selected decoding, attention, and cache routes,
cache allocation, TTFT, inter-token latency, end-to-end latency, output tokens
per second, percentile summaries, and the model/device/dtype/software metadata
needed to interpret them.

`none` recomputes the whole sequence for every new token and is the correctness
baseline. `contiguous` preallocates a request-local cache and is the default.
`paged` allocates fixed-size blocks lazily, but currently gathers them before
eager attention; it teaches paging and is not yet a PagedAttention speedup.

For comparable decode experiments, save one aggregate result per configuration
to the local cache, then regenerate the tracked leaderboard. Run these
from the repository root, one at a time:

```bash
tinyinfer bench --profile decode --kv-cache none --save
tinyinfer bench --profile decode --kv-cache contiguous --save
tinyinfer leaderboard
```

Of the end-to-end benchmark outputs, only `BENCHMARKS.md` belongs in Git.
TinyInfer does not retain prompts, generated text, or individual runs when
`--save` is used.

Run the focused Q8/BF16 Roofline experiment with:

```bash
tinyinfer benchmark --roofline
tinyinfer benchmark --roofline --capture
tinyinfer benchmark --roofline --clean
```

Generated graphs, results, and traces stay in the ignored
`.tinyinfer/roofline/` directory.

---

## Development

```bash
python -m pytest -q
ruff check .
ruff format --check .
```
