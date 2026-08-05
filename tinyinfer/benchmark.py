from __future__ import annotations

import platform
import statistics
import time
from itertools import pairwise
from typing import Any

import torch

from tinyinfer.engine import Engine
from tinyinfer.tokenizer import Message


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
    end_to_end = finished - started
    return {
        "time_to_first_token_seconds": time_to_first,
        "inter_token_latency_seconds": inter_token,
        "end_to_end_latency_seconds": end_to_end,
        "output_tokens": token_count,
        "output_tokens_per_second": token_count / end_to_end if end_to_end else 0.0,
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
        ]
    )
