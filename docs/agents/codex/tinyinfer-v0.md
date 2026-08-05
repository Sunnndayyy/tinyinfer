# TinyInfer V0 fast paths

- Runtime direction: `server -> engine -> model`; artifacts/tokenizer are leaf helpers.
- Real smoke: `.venv/bin/tinyinfer generate Qwen/Qwen2.5-1.5B-Instruct --prompt 'What does a KV cache save?' --max-new-tokens 8 --device mps`.
- Network smoke: serve on `127.0.0.1:8123`, then call `tinyinfer chat --host http://127.0.0.1:8123`.
- Fast verification: `.venv/bin/python -m pytest -q` before loading the 3 GB checkpoint.
- Qwen checkpoint keys match the module path rooted at `model.*`; tied logits use `model.embed_tokens.weight` directly.
- Construct the full model on `meta`, assign safetensor shards, then move once to MPS to avoid random 1.5B-weight initialization.
- Gotcha: V0 intentionally has no KV cache, so every generated token performs a full forward pass over the growing sequence.
- Keep request validation outside the token generator: generator bodies are lazy, so errors raised inside them happen after an SSE response may have committed HTTP 200.
- A streaming response owns the single model slot until its complete ASGI lifecycle ends; release it around `StreamingResponse.__call__`, not only inside the body iterator.
- Preserve the generator return value (`stop` vs `length`) when converting token events into API responses.
