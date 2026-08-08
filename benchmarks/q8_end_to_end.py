"""Sweep exact input-token lengths through matched BF16 and Q8 models on MPS."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from tinyinfer.artifacts import resolve_model
from tinyinfer.benchmark import hardware_name
from tinyinfer.engine import Engine
from tinyinfer.kv_cache import create_kv_cache

LENGTHS = (1, 8, 32, 128, 512)
SEED_TEXT = "TinyInfer measures prompt processing with exact token counts. "


@dataclass(frozen=True)
class Sample:
    first_forward_seconds: float
    ttft_seconds: float
    decode_tokens_per_second: float
    output_tokens: int


def sample_order(sample_index: int) -> tuple[str, str]:
    """Alternate which format runs first to reduce fixed ordering effects."""
    return ("bf16", "q8") if sample_index % 2 == 0 else ("q8", "bf16")


def exact_prompt_ids(seed_ids: list[int], length: int) -> list[int]:
    if length < 1:
        raise ValueError("input token length must be at least 1")
    if not seed_ids:
        raise ValueError("seed text must encode to at least one token")
    repetitions = (length + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repetitions)[:length]


def create_cache(engine: Engine, *, prompt_length: int, max_new_tokens: int):
    config = engine.model.config
    return create_kv_cache(
        engine.kv_cache_name,
        num_layers=config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        capacity=prompt_length + max_new_tokens - 1,
        block_size=engine.kv_cache_block_size,
        device=engine.device,
        dtype=engine.activation_dtype,
    )


def measure_first_forward(
    engine: Engine,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
) -> float:
    input_ids = torch.tensor([prompt_ids], device=engine.device, dtype=torch.long)
    cache = create_cache(
        engine,
        prompt_length=len(prompt_ids),
        max_new_tokens=max_new_tokens,
    )
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = engine.model.next_token_logits(input_ids, cache=cache)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    del logits, cache, input_ids
    return elapsed


def measure_generation(
    engine: Engine,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
) -> tuple[float, float, int]:
    torch.mps.synchronize()
    started = time.perf_counter()
    arrival_times = [
        time.perf_counter()
        for _event in engine.decoder.stream(prompt_ids, max_new_tokens=max_new_tokens)
    ]
    torch.mps.synchronize()
    finished = time.perf_counter()

    time_to_first = arrival_times[0] - started if arrival_times else finished - started
    decode_time = arrival_times[-1] - arrival_times[0] if len(arrival_times) > 1 else 0.0
    decode_tps = (len(arrival_times) - 1) / decode_time if decode_time else 0.0
    return time_to_first, decode_tps, len(arrival_times)


def measure_sample(
    engine: Engine,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
) -> Sample:
    first_forward = measure_first_forward(
        engine,
        prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    ttft, decode_tps, output_tokens = measure_generation(
        engine,
        prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    return Sample(first_forward, ttft, decode_tps, output_tokens)


def median_sample(samples: list[Sample]) -> Sample:
    return Sample(
        first_forward_seconds=statistics.median(sample.first_forward_seconds for sample in samples),
        ttft_seconds=statistics.median(sample.ttft_seconds for sample in samples),
        decode_tokens_per_second=statistics.median(
            sample.decode_tokens_per_second for sample in samples
        ),
        output_tokens=int(statistics.median(sample.output_tokens for sample in samples)),
    )


def measure_complete_sample(
    name: str,
    engine: Engine,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
) -> Sample:
    sample = measure_sample(engine, prompt_ids, max_new_tokens=max_new_tokens)
    if sample.output_tokens != max_new_tokens:
        raise RuntimeError(
            f"{name} generated {sample.output_tokens} of {max_new_tokens} requested tokens; "
            "choose a prompt whose BF16 and Q8 runs both reach the limit"
        )
    return sample


def measure_pair(
    engines: dict[str, Engine],
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Sample]:
    for sample_index in range(warmup):
        for name in sample_order(sample_index):
            measure_complete_sample(
                name,
                engines[name],
                prompt_ids,
                max_new_tokens=max_new_tokens,
            )

    samples: dict[str, list[Sample]] = {"bf16": [], "q8": []}
    for sample_index in range(repetitions):
        for name in sample_order(sample_index):
            samples[name].append(
                measure_complete_sample(
                    name,
                    engines[name],
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                )
            )
    return {name: median_sample(values) for name, values in samples.items()}


def load_engine(model: str, *, quantization: str, attention: str, kv_cache: str) -> Engine:
    model_path = resolve_model(model)
    return Engine.from_pretrained(
        model_path,
        device_name="mps",
        dtype_name="bfloat16",
        attention_name=attention,
        kv_cache_name=kv_cache,
        quantization_name=quantization,
    )


def validate_matched_engines(engines: dict[str, Engine], seed_text: str) -> list[int]:
    bf16 = engines["bf16"]
    q8 = engines["q8"]
    if bf16.quantization_name != "none" or q8.quantization_name != "q8":
        raise ValueError("expected one BF16 checkpoint and one Q8 checkpoint")
    if bf16.model.config != q8.model.config:
        raise ValueError("BF16 and Q8 model configurations do not match")
    if not bf16.source_revision or not q8.source_revision:
        raise ValueError("BF16 and Q8 checkpoints must expose a source revision")
    if bf16.source_revision != q8.source_revision:
        raise ValueError("BF16 and Q8 source revisions do not match")
    runtime = ("device", "activation_dtype", "attention_name", "kv_cache_name")
    for attribute in runtime:
        if getattr(bf16, attribute) != getattr(q8, attribute):
            raise ValueError(f"BF16 and Q8 {attribute} values do not match")
    bf16_seed = bf16.tokenizer.encode(seed_text)
    if bf16_seed != q8.tokenizer.encode(seed_text):
        raise ValueError("BF16 and Q8 tokenizers do not match")
    return bf16_seed


def model_label(engine: Engine) -> str:
    revision = engine.source_revision or "unknown revision"
    return f"{engine.source_model} ({revision})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bf16_model",
        help="versioned BF16 Hugging Face model id or snapshot directory",
    )
    parser.add_argument("q8_model", help="TinyInfer Q8 model directory")
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--kv-cache", choices=("contiguous", "paged"), default="contiguous")
    args = parser.parse_args()
    if (
        any(length < 1 for length in args.lengths)
        or args.max_new_tokens < 2
        or args.warmup < 0
        or args.repetitions < 1
    ):
        parser.error("lengths must be >= 1, max-new-tokens >= 2, warmup >= 0, and repetitions >= 1")

    engines = {
        "bf16": load_engine(
            args.bf16_model,
            quantization="none",
            attention=args.attention,
            kv_cache=args.kv_cache,
        ),
        "q8": load_engine(
            args.q8_model,
            quantization="q8",
            attention=args.attention,
            kv_cache=args.kv_cache,
        ),
    }
    seed_ids = validate_matched_engines(engines, SEED_TEXT)

    print(f"hardware: {hardware_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"BF16: {model_label(engines['bf16'])}")
    print(f"Q8: {model_label(engines['q8'])}")
    print(f"attention: {args.attention}; KV cache: {args.kv_cache}")
    print(
        f"warmup: {args.warmup}; repetitions: {args.repetitions}; "
        f"max new tokens: {args.max_new_tokens}"
    )
    print("TTFT starts from exact token IDs, so tokenizer time is excluded.")
    print("tokens   first forward ms       TTFT ms              decode tok/s       Q8/BF16 decode")
    print("         BF16       Q8         BF16       Q8         BF16       Q8")
    for length in args.lengths:
        prompt_ids = exact_prompt_ids(seed_ids, length)
        results = measure_pair(
            engines,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        bf16 = results["bf16"]
        q8 = results["q8"]
        print(
            f"{length:>6}  "
            f"{bf16.first_forward_seconds * 1000:>8.2f} {q8.first_forward_seconds * 1000:>8.2f}  "
            f"{bf16.ttft_seconds * 1000:>8.2f} {q8.ttft_seconds * 1000:>8.2f}  "
            f"{bf16.decode_tokens_per_second:>8.2f} {q8.decode_tokens_per_second:>8.2f}  "
            f"{q8.decode_tokens_per_second / bf16.decode_tokens_per_second:>8.3f}x"
        )


if __name__ == "__main__":
    main()
