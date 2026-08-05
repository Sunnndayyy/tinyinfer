from tinyinfer.benchmark import distribution, format_summary, percentile


def test_percentile_interpolates_a_small_sample() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == 3.8499999999999996


def test_empty_distribution_is_zero_instead_of_crashing() -> None:
    assert distribution([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_summary_records_the_selected_decoding_implementation() -> None:
    distribution_values = {"p50": 0.1, "p95": 0.2, "p99": 0.3}
    result = {
        "metadata": {
            "model": "test-model",
            "device": "cpu",
            "dtype": "float32",
            "decoding": "autoregressive",
            "kv_cache": "contiguous",
            "repetitions": 1,
            "warmup": 0,
        },
        "metrics": {
            "time_to_first_token_seconds": distribution_values,
            "inter_token_latency_seconds": distribution_values,
            "end_to_end_latency_seconds": distribution_values,
            "output_tokens_per_second": {**distribution_values, "mean": 10.0},
            "kv_cache_bytes": 128,
        },
    }

    summary = format_summary(result)

    assert "decoding: autoregressive" in summary
