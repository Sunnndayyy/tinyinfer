# TinyInfer

TinyInfer is a small, readable LLM inference server.
It aims for the space between learning basic concepts and a production engine such as vLLM, SGlang etc. 

The first model target is
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
on Apple Silicon. TinyInfer downloads Qwen's tokenizer and safetensors.

## overview

Optimization routes stay explicit so each faster implementation can be checked
against a readable reference path.

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

Convert it to TinyInfer's Q8 Safetensors format:

```bash
tinyinfer quantize Qwen/Qwen2.5-1.5B-Instruct \
  --format q8 \
  --output .tinyinfer/models/qwen2.5-1.5b-q8
```

The Q8 directory is self-contained and keeps the tied embedding/output matrix
once. On Apple Silicon, packed weights and FP16 scales are read directly by a
fused Metal linear kernel; CPU keeps the deliberately slow reference path.

```bash
tinyinfer generate .tinyinfer/models/qwen2.5-1.5b-q8 \
  --device mps \
  --dtype bfloat16 \
  --prompt "What does weight quantization save?" \
  --max-new-tokens 8
```

Take the model for a spin:

```bash
tinyinfer generate Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "What does a KV cache save?" \
  --decoding autoregressive \
  --attention eager \
  --kv-cache contiguous \
  --max-new-tokens 24
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
to the ignored local cache, then regenerate the tracked leaderboard. Run these
from the repository root, one at a time:

```bash
tinyinfer bench --profile decode --kv-cache none --save
tinyinfer bench --profile decode --kv-cache contiguous --save
tinyinfer leaderboard
```

Of the end-to-end benchmark outputs, only `BENCHMARKS.md` belongs in Git.
TinyInfer does not retain prompts, generated text, or individual runs when
`--save` is used.

The fused Q8 operator has a separate warmed MPS benchmark for the two Qwen
decode shapes used as its promotion gate:

```bash
.venv/bin/python benchmarks/q8_mps_linear.py
```

See the [M4 Pro Q8 benchmark note](benchmarks/README.md) for the
retained operator, decode-throughput, and TTFT results. TTFT is reported
separately because Q8 speeds up steady-state decode while regressing prefill.

## Current limitations

- Qwen2 only; V0 is checked against Qwen2.5-1.5B-Instruct.
- Greedy decoding only.
- Batch size one and one in-process model.
- Request-local contiguous and educational paged KV caches; no shared cache pool.
- Selectable eager and PyTorch SDPA attention; no FlashAttention-specific route yet.
- BF16 MPS by default on Apple Silicon; CPU uses float32.
- Q8 conversion, a readable CPU oracle, and fused MPS execution.
- No request queue, auth, Linux/CUDA, or distributed execution.

## Development

```bash
python -m pytest -q
ruff check .
ruff format --check .
```
