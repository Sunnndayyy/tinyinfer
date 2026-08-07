import json
from types import SimpleNamespace

import pytest

from tinyinfer import benchmark as benchmark_module
from tinyinfer.benchmark import (
    benchmark_options,
    distribution,
    format_leaderboard,
    format_summary,
    load_records,
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


def test_summary_records_the_selected_runtime_implementations() -> None:
    distribution_values = {"p50": 0.1, "p95": 0.2, "p99": 0.3}
    result = {
        "metadata": {
            "model": "test-model",
            "device": "cpu",
            "dtype": "float32",
            "quantization": "q8",
            "decoding": "autoregressive",
            "attention": "sdpa",
            "kv_cache": "contiguous",
            "repetitions": 1,
            "warmup": 0,
        },
        "metrics": {
            "time_to_first_token_seconds": distribution_values,
            "inter_token_latency_seconds": distribution_values,
            "end_to_end_latency_seconds": distribution_values,
            "output_tokens_per_second": {**distribution_values, "mean": 10.0},
            "decode_tokens_per_second": distribution_values,
            "kv_cache_bytes": 128,
        },
    }

    summary = format_summary(result)

    assert "decoding: autoregressive" in summary
    assert "attention: sdpa" in summary
    assert "weights: q8" in summary

    result["metadata"].pop("attention")

    assert "attention: unknown" in format_summary(result)


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


def leaderboard_result(
    cache: str,
    decode_tps: float,
    attention: str = "eager",
    *,
    quantization: str = "none",
    revision: str = "revision-1",
    artifact_path: str = "/models/qwen",
) -> dict:
    return {
        "metadata": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "hardware": "Apple M3",
            "device": "mps",
            "dtype": "torch.bfloat16",
            "quantization": quantization,
            "revision": revision,
            "artifact_path": artifact_path,
            "profile": "decode",
            "decoding": "autoregressive",
            "attention": attention,
            "kv_cache": cache,
        },
        "metrics": {
            "time_to_first_token_seconds": {"p50": 0.4},
            "decode_tokens_per_second": {"p50": decode_tps},
        },
        "runs": [{"large": "raw sample"}],
    }


def leaderboard_row(
    cache: str,
    decode_tps: float,
    quantization: str = "none",
    *,
    revision: str = "revision-1",
) -> dict:
    result = leaderboard_result(
        cache,
        decode_tps,
        quantization=quantization,
        revision=revision,
    )
    return {
        **result["metadata"],
        "ttft_seconds": 0.4,
        "decode_tokens_per_second": decode_tps,
    }


def test_saved_results_upsert_one_summary_per_configuration(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"

    save_leaderboard_result(leaderboard_result("none", 10.0), store)
    save_leaderboard_result(leaderboard_result("contiguous", 20.0), store)
    save_leaderboard_result(leaderboard_result("contiguous", 22.0), store)
    save_leaderboard_result(leaderboard_result("contiguous", 30.0, "sdpa"), store)

    saved = json.loads(store.read_text())
    assert len(saved) == 3
    assert [row["decode_tokens_per_second"] for row in saved] == [10.0, 22.0, 30.0]
    assert all("runs" not in row for row in saved)


def test_saved_weight_formats_coexist_but_artifact_paths_do_not_split_identity(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"

    save_leaderboard_result(leaderboard_result("contiguous", 20.0), store)
    save_leaderboard_result(leaderboard_result("contiguous", 30.0, quantization="q8"), store)
    save_leaderboard_result(leaderboard_result("contiguous", 40.0, quantization="q4"), store)
    save_leaderboard_result(
        leaderboard_result(
            "contiguous",
            31.0,
            quantization="q8",
            artifact_path="/models/another-q8-copy",
        ),
        store,
    )

    saved = load_records(store)
    assert len(saved) == 3
    assert {row["quantization"] for row in saved} == {"none", "q8", "q4"}
    q8 = next(row for row in saved if row["quantization"] == "q8")
    assert q8["decode_tokens_per_second"] == 31.0
    assert q8["artifact_path"] == "/models/another-q8-copy"


def test_saved_revisions_remain_separate_configurations(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"

    save_leaderboard_result(leaderboard_result("contiguous", 20.0), store)
    save_leaderboard_result(
        leaderboard_result("contiguous", 30.0, revision="revision-2"),
        store,
    )

    saved = load_records(store)
    assert len(saved) == 2
    assert {(row["revision"], row["decode_tokens_per_second"]) for row in saved} == {
        ("revision-1", 20.0),
        ("revision-2", 30.0),
    }


def test_corrupt_local_results_have_an_actionable_error(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"
    store.write_text("not json")

    with pytest.raises(ValueError, match="delete it and rerun"):
        save_leaderboard_result(leaderboard_result("none", 10.0), store)


def test_legacy_saved_results_default_to_autoregressive(tmp_path) -> None:
    store = tmp_path / "benchmarks.json"
    legacy = leaderboard_result("none", 10.0)
    del legacy["metadata"]["decoding"]
    store.write_text(
        json.dumps(
            [
                {
                    "model": legacy["metadata"]["model"],
                    "hardware": legacy["metadata"]["hardware"],
                    "device": legacy["metadata"]["device"],
                    "dtype": legacy["metadata"]["dtype"],
                    "profile": legacy["metadata"]["profile"],
                    "kv_cache": legacy["metadata"]["kv_cache"],
                    "ttft_seconds": 0.4,
                    "decode_tokens_per_second": 10.0,
                }
            ]
        )
    )

    record = load_records(store)[0]
    assert record["decoding"] == "autoregressive"
    assert record["attention"] == "unknown"
    assert record["quantization"] == "none"
    assert record["revision"] is None
    assert record["artifact_path"] is None


def test_leaderboard_keeps_cache_and_weight_format_comparisons_separate() -> None:
    rows = [
        leaderboard_row("none", 10.0),
        leaderboard_row("contiguous", 20.0),
        leaderboard_row("none", 15.0, "q8"),
        leaderboard_row("contiguous", 30.0, "q8"),
        leaderboard_row("contiguous", 40.0, "q4"),
        leaderboard_row("paged", 45.0, "q8"),
        leaderboard_row("contiguous", 25.0, "q8", revision="revision-2"),
    ]

    markdown = format_leaderboard(rows)

    assert "| revision-1 |" in markdown
    assert "| revision-2 |" in markdown
    assert (
        "| autoregressive | eager | none | none | 0.400s | 10.0 | baseline | baseline |" in markdown
    )
    assert (
        "| autoregressive | eager | none | contiguous | 0.400s | 20.0 | +100.0% | baseline |"
        in markdown
    )
    assert "| autoregressive | eager | q8 | none | 0.400s | 15.0 | baseline | +50.0% |" in markdown
    assert (
        "| autoregressive | eager | q8 | contiguous | 0.400s | 30.0 | +100.0% | +50.0% |"
        in markdown
    )
    assert "| autoregressive | eager | q4 | contiguous | 0.400s | 40.0 | — | +100.0% |" in markdown
    assert "| autoregressive | eager | q8 | paged | 0.400s | 45.0 | +200.0% | — |" in markdown
    assert "| autoregressive | eager | q8 | contiguous | 0.400s | 25.0 | — | — |" in markdown


def test_saving_requires_a_named_profile(tmp_path) -> None:
    result = leaderboard_result("none", 10.0)
    result["metadata"]["profile"] = None

    with pytest.raises(ValueError, match="profile"):
        save_leaderboard_result(result, tmp_path / "benchmarks.json")
