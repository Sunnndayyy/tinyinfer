from tinyinfer.benchmark import distribution, percentile


def test_percentile_interpolates_a_small_sample() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == 3.8499999999999996


def test_empty_distribution_is_zero_instead_of_crashing() -> None:
    assert distribution([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
