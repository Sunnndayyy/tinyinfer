from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from tinyinfer.engine import Engine
from tinyinfer.runtime import KV_CACHE_NAMES
from tinyinfer.tokenizer import Message

LOCAL_RESULTS = Path(".tinyinfer/benchmarks.json")
LEADERBOARD_PATH = Path("BENCHMARKS.md")
RESULT_IDENTITY_FIELDS = (
    "model",
    "hardware",
    "device",
    "dtype",
    "profile",
    "decoding",
    "kv_cache",
)
COMPARISON_GROUP_FIELDS = RESULT_IDENTITY_FIELDS[:-1]
RESULT_FIELDS = (*RESULT_IDENTITY_FIELDS, "ttft_seconds", "decode_tokens_per_second")
LEGACY_RESULT_FIELDS = tuple(field for field in RESULT_FIELDS if field != "decoding")


@dataclass(frozen=True)
class BenchmarkOptions:
    prompt: str
    system: str
    max_new_tokens: int
    warmup: int
    repetitions: int
    profile: str | None = None


DEFAULT_BENCHMARK = BenchmarkOptions(
    prompt="Explain what a KV cache saves in one sentence.",
    system="You are a helpful assistant.",
    max_new_tokens=16,
    warmup=1,
    repetitions=3,
)

PROFILES = {
    "decode": BenchmarkOptions(
        prompt="Explain how transformer attention works in detail.",
        system="You are a helpful assistant.",
        max_new_tokens=64,
        warmup=2,
        repetitions=5,
        profile="decode",
    )
}


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def benchmark_options(
    profile: str | None,
    *,
    prompt: str | None,
    system: str | None,
    max_new_tokens: int | None,
    warmup: int | None,
    repetitions: int | None,
) -> BenchmarkOptions:
    overrides = (prompt, system, max_new_tokens, warmup, repetitions)
    if profile:
        if any(value is not None for value in overrides):
            raise ValueError("--profile cannot be combined with prompt or run-size options")
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from: {', '.join(PROFILES)}")
        return PROFILES[profile]
    return BenchmarkOptions(
        prompt=DEFAULT_BENCHMARK.prompt if prompt is None else prompt,
        system=DEFAULT_BENCHMARK.system if system is None else system,
        max_new_tokens=(
            DEFAULT_BENCHMARK.max_new_tokens if max_new_tokens is None else max_new_tokens
        ),
        warmup=DEFAULT_BENCHMARK.warmup if warmup is None else warmup,
        repetitions=DEFAULT_BENCHMARK.repetitions if repetitions is None else repetitions,
    )


def hardware_name() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ("sysctl", "-n", "machdep.cpu.brand_string"),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def run_once(
    engine: Engine,
    messages: list[Message],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    arrival_times: list[float] = []
    output_parts: list[str] = []
    for event in engine.stream(messages, max_new_tokens=max_new_tokens):
        arrival_times.append(time.perf_counter())
        output_parts.append(event.text)
    finished = time.perf_counter()

    token_count = len(arrival_times)
    time_to_first = arrival_times[0] - started if arrival_times else finished - started
    inter_token = [later - earlier for earlier, later in pairwise(arrival_times)]
    decode_time = arrival_times[-1] - arrival_times[0] if token_count > 1 else 0.0
    end_to_end = finished - started
    return {
        "time_to_first_token_seconds": time_to_first,
        "inter_token_latency_seconds": inter_token,
        "end_to_end_latency_seconds": end_to_end,
        "output_tokens": token_count,
        "output_tokens_per_second": token_count / end_to_end if end_to_end else 0.0,
        "decode_tokens_per_second": (token_count - 1) / decode_time if decode_time else 0.0,
        "kv_cache_bytes": engine.last_cache_bytes,
        "text": "".join(output_parts),
    }


def benchmark(
    engine: Engine,
    messages: list[Message],
    *,
    max_new_tokens: int,
    warmup: int,
    repetitions: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    for _ in range(warmup):
        run_once(engine, messages, max_new_tokens=max_new_tokens)

    runs = [run_once(engine, messages, max_new_tokens=max_new_tokens) for _ in range(repetitions)]
    time_to_first = [run["time_to_first_token_seconds"] for run in runs]
    end_to_end = [run["end_to_end_latency_seconds"] for run in runs]
    throughput = [run["output_tokens_per_second"] for run in runs]
    decode_throughput = [run["decode_tokens_per_second"] for run in runs]
    inter_token = [latency for run in runs for latency in run["inter_token_latency_seconds"]]
    cache_bytes = [run["kv_cache_bytes"] for run in runs]
    return {
        "metadata": {
            **metadata,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "metrics": {
            "time_to_first_token_seconds": distribution(time_to_first),
            "inter_token_latency_seconds": distribution(inter_token),
            "end_to_end_latency_seconds": distribution(end_to_end),
            "output_tokens_per_second": {
                **distribution(throughput),
                "mean": statistics.fmean(throughput),
            },
            "decode_tokens_per_second": distribution(decode_throughput),
            "kv_cache_bytes": max(cache_bytes, default=0),
        },
        "runs": runs,
    }


def format_summary(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    metrics = result["metrics"]
    ttft = metrics["time_to_first_token_seconds"]
    itl = metrics["inter_token_latency_seconds"]
    e2e = metrics["end_to_end_latency_seconds"]
    throughput = metrics["output_tokens_per_second"]
    decode_throughput = metrics["decode_tokens_per_second"]
    return "\n".join(
        [
            f"model: {metadata['model']}",
            f"device: {metadata['device']} ({metadata['dtype']})",
            f"decoding: {metadata['decoding']}",
            f"KV cache: {metadata['kv_cache']} ({metrics['kv_cache_bytes']} bytes)",
            f"runs: {metadata['repetitions']} after {metadata['warmup']} warmup",
            f"TTFT seconds       p50 {ttft['p50']:.3f} | p95 {ttft['p95']:.3f} | p99 {ttft['p99']:.3f}",
            f"inter-token sec    p50 {itl['p50']:.3f} | p95 {itl['p95']:.3f} | p99 {itl['p99']:.3f}",
            f"end-to-end sec     p50 {e2e['p50']:.3f} | p95 {e2e['p95']:.3f} | p99 {e2e['p99']:.3f}",
            f"output tokens/sec  mean {throughput['mean']:.2f} | p50 {throughput['p50']:.2f}",
            f"decode tokens/sec  p50 {decode_throughput['p50']:.2f}",
        ]
    )


def save_leaderboard_result(result: dict[str, Any], path: Path = LOCAL_RESULTS) -> dict[str, Any]:
    metadata = result["metadata"]
    if not metadata.get("profile"):
        raise ValueError("saving a leaderboard result requires --profile")
    record = {
        "model": metadata["model"],
        "hardware": metadata["hardware"],
        "device": metadata["device"],
        "dtype": metadata["dtype"],
        "profile": metadata["profile"],
        "decoding": metadata["decoding"],
        "kv_cache": metadata["kv_cache"],
        "ttft_seconds": result["metrics"]["time_to_first_token_seconds"]["p50"],
        "decode_tokens_per_second": result["metrics"]["decode_tokens_per_second"]["p50"],
    }
    records = load_records(path)
    identity = record_key(record, RESULT_IDENTITY_FIELDS)
    records = [saved for saved in records if record_key(saved, RESULT_IDENTITY_FIELDS) != identity]
    records.append(record)
    write_records(path, records)
    return record


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}; delete it and rerun the benchmark") from error
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or not set(LEGACY_RESULT_FIELDS) <= record.keys()
        for record in records
    ):
        raise ValueError(f"invalid benchmark data in {path}; delete it and rerun")
    for record in records:
        record.setdefault("decoding", "autoregressive")
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(path)
    except OSError as error:
        raise ValueError(f"cannot save benchmark data to {path}") from error


def record_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(record[field] for field in fields)


def format_leaderboard(records: list[dict[str, Any]]) -> str:
    records = [{"decoding": "autoregressive", **record} for record in records]
    groups = {}
    for record in records:
        group = record_key(record, COMPARISON_GROUP_FIELDS)
        if record["kv_cache"] == "none":
            groups[group] = record["decode_tokens_per_second"]

    lines = [
        "# TinyInfer benchmark leaderboard",
        "",
        "Generated by `tinyinfer leaderboard` from local aggregate results. Compare rows only within the same model, hardware, dtype, profile, and decoder.",
        "",
        "| Model | Hardware | Profile | Decoder | KV cache | TTFT p50 | Decode tok/s | Change |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    cache_order = {name: position for position, name in enumerate(KV_CACHE_NAMES)}
    for record in sorted(
        records,
        key=lambda item: (
            *tuple(str(value) for value in record_key(item, COMPARISON_GROUP_FIELDS)),
            cache_order.get(item["kv_cache"], 99),
        ),
    ):
        group = record_key(record, COMPARISON_GROUP_FIELDS)
        baseline = groups.get(group)
        if record["kv_cache"] == "none":
            change = "baseline"
        elif baseline:
            change = f"{(record['decode_tokens_per_second'] / baseline - 1) * 100:+.1f}%"
        else:
            change = "—"
        hardware = f"{record['hardware']} · {record['device']}/{str(record['dtype']).removeprefix('torch.')}"
        lines.append(
            f"| {record['model']} | {hardware} | {record['profile']} | {record['decoding']} | "
            f"{record['kv_cache']} | "
            f"{record['ttft_seconds']:.3f}s | {record['decode_tokens_per_second']:.1f} | {change} |"
        )
    return "\n".join(lines) + "\n"


def write_leaderboard(store: Path = LOCAL_RESULTS, output: Path = LEADERBOARD_PATH) -> str:
    if not store.exists():
        raise ValueError("no saved benchmark results; run `tinyinfer bench --profile ... --save`")
    markdown = format_leaderboard(load_records(store))
    if not output.exists() or output.read_text() != markdown:
        output.write_text(markdown)
    return markdown
