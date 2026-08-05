from tinyinfer.benchmark import distribution, format_summary, percentile


def test_percentile_interpolates_a_small_sample() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == 3.8499999999999996


def test_empty_distribution_is_zero_instead_of_crashing() -> None:
    assert distribution([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_summary_records_the_attention_implementation() -> None:
    result = {
        "metadata": {
            "model": "tiny",
            "device": "cpu",
            "dtype": "torch.float32",
            "attention": "sdpa",
            "kv_cache": "contiguous",
            "repetitions": 1,
            "warmup": 0,
        },
        "metrics": {
            "time_to_first_token_seconds": {"p50": 0.1, "p95": 0.1, "p99": 0.1},
            "inter_token_latency_seconds": {"p50": 0.1, "p95": 0.1, "p99": 0.1},
            "end_to_end_latency_seconds": {"p50": 0.1, "p95": 0.1, "p99": 0.1},
            "output_tokens_per_second": {"mean": 10.0, "p50": 10.0},
            "kv_cache_bytes": 16,
        },
    }

    assert "attention: sdpa" in format_summary(result)

    result["metadata"].pop("attention")

    assert "attention: unknown" in format_summary(result)
