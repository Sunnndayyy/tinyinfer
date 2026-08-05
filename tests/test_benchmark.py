import json
from types import SimpleNamespace

import pytest

from tinyinfer import benchmark as benchmark_module
from tinyinfer.benchmark import (
    benchmark_options,
    distribution,
    format_leaderboard,
    percentile,
    run_once,
    save_leaderboard_result,
)


def test_percentile_interpolates_a_small_sample() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == 3.8499999999999996


def test_empty_distribution_is_zero_instead_of_crashing() -> None:
    assert distribution([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_decode_throughput_excludes_time_to_first_token(monkeypatch) -> None:
    clock = iter((0.0, 2.0, 3.0, 4.0, 5.0))
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(clock))
    engine = SimpleNamespace(
        last_cache_bytes=64,
        stream=lambda messages, max_new_tokens: iter(
            (SimpleNamespace(text="a"), SimpleNamespace(text="b"), SimpleNamespace(text="c"))
        ),
    )

    result = run_once(engine, [], max_new_tokens=3)

    assert result["time_to_first_token_seconds"] == 2.0
    assert result["decode_tokens_per_second"] == 1.0
    assert result["output_tokens_per_second"] == 0.6


def test_benchmark_options_preserve_explicit_empty_strings() -> None:
    options = benchmark_options(
        None,
        prompt="",
        system="",
        max_new_tokens=None,
        warmup=None,
        repetitions=None,
    )

    assert options.prompt == ""
    assert options.system == ""


def test_unknown_benchmark_profile_is_actionable() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        benchmark_options(
            "missing",
            prompt=None,
            system=None,
            max_new_tokens=None,
            warmup=None,
            repetitions=None,
        )


def leaderboard_result(cache: str, decode_tps: float) -> dict:
    return {
        "metadata": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "hardware": "Apple M3",
            "device": "mps",
            "dtype": "torch.bfloat16",
            "profile": "decode",
            "kv_cache": cache,
        },
        "metrics": {
            "time_to_first_token_seconds": {"p50": 0.4},
            "decode_tokens_per_second": {"p50": decode_tps},
        },
        "runs": [{"large": "raw sample"}],
    }


def test_saved_results_upsert_one_summary_per_configuration(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"

    save_leaderboard_result(leaderboard_result("none", 10.0), store)
    save_leaderboard_result(leaderboard_result("contiguous", 20.0), store)
    save_leaderboard_result(leaderboard_result("contiguous", 22.0), store)

    saved = json.loads(store.read_text())
    assert len(saved) == 2
    assert [row["decode_tokens_per_second"] for row in saved] == [10.0, 22.0]
    assert all("runs" not in row for row in saved)


def test_corrupt_local_results_have_an_actionable_error(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"
    store.write_text("not json")

    with pytest.raises(ValueError, match="delete it and rerun"):
        save_leaderboard_result(leaderboard_result("none", 10.0), store)


def test_leaderboard_compares_decode_speed_with_uncached_baseline() -> None:
    rows = [
        {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "hardware": "Apple M3",
            "device": "mps",
            "dtype": "torch.bfloat16",
            "profile": "decode",
            "kv_cache": "none",
            "ttft_seconds": 0.4,
            "decode_tokens_per_second": 10.0,
        },
        {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "hardware": "Apple M3",
            "device": "mps",
            "dtype": "torch.bfloat16",
            "profile": "decode",
            "kv_cache": "contiguous",
            "ttft_seconds": 0.4,
            "decode_tokens_per_second": 20.0,
        },
    ]

    markdown = format_leaderboard(rows)

    assert "| none | 0.400s | 10.0 | baseline |" in markdown
    assert "| contiguous | 0.400s | 20.0 | +100.0% |" in markdown


def test_saving_requires_a_named_profile(tmp_path) -> None:
    result = leaderboard_result("none", 10.0)
    result["metadata"]["profile"] = None

    with pytest.raises(ValueError, match="profile"):
        save_leaderboard_result(result, tmp_path / "benchmarks.json")
