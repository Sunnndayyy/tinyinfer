import sys
from contextlib import nullcontext

import pytest
import torch

from benchmarks import q8_mps_linear as benchmark

QWEN_SHAPES = benchmark.QWEN_SHAPES
operation_counts = benchmark.operation_counts
profile_operation = benchmark.profile_operation


def test_roofline_counts_expose_the_minimum_bf16_and_q8_data_movement() -> None:
    bf16 = operation_counts("bf16", rows=1, input_width=32, output_width=4)
    q8 = operation_counts("q8", rows=1, input_width=32, output_width=4)

    assert bf16.flops == q8.flops == 256
    assert bf16.bytes == 64 + 256 + 8
    assert q8.bytes == 64 + 128 + 8 + 8
    assert q8.arithmetic_intensity > bf16.arithmetic_intensity


def test_roofline_intensity_rises_when_rows_reuse_the_same_weights() -> None:
    input_width, output_width = QWEN_SHAPES["mlp-up"]
    decode = operation_counts("bf16", rows=1, input_width=input_width, output_width=output_width)
    prefill = operation_counts("bf16", rows=64, input_width=input_width, output_width=output_width)

    assert prefill.flops == 64 * decode.flops
    assert prefill.arithmetic_intensity > 50 * decode.arithmetic_intensity


def test_profile_operation_separates_warmup_from_capture(monkeypatch) -> None:
    calls = {"operation": 0, "synchronize": 0}
    profiler_arguments = []

    def operation() -> None:
        calls["operation"] += 1

    def synchronize() -> None:
        calls["synchronize"] += 1

    def profile(mode, *, wait_until_completed):
        profiler_arguments.append((mode, wait_until_completed))
        return nullcontext()

    monkeypatch.setattr("torch.mps.synchronize", synchronize)
    monkeypatch.setattr("torch.mps.profiler.profile", profile)

    profile_operation(
        operation,
        warmup=2,
        iterations=3,
    )

    assert calls == {"operation": 5, "synchronize": 2}
    assert profiler_arguments == [("interval,event", True)]


@pytest.mark.parametrize(
    ("path", "expected_operation"),
    (("q8", benchmark._q8_linear_mps), ("bf16", benchmark.F.linear)),
)
def test_main_routes_profile_mode_without_running_benchmark(
    monkeypatch, path, expected_operation
) -> None:
    captured = {}

    monkeypatch.setattr(sys, "argv", ["q8_mps_linear.py", "--rows", "1", "--profile-path", path])
    monkeypatch.setattr(benchmark, "require_q8_mps", lambda: None)
    monkeypatch.setattr(benchmark, "hardware_name", lambda: "test GPU")
    monkeypatch.setattr(benchmark.torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(
        benchmark.torch,
        "randn",
        lambda shape, **_kwargs: torch.empty(shape),
    )
    monkeypatch.setattr(benchmark, "make_q8_weights", lambda *_widths: (object(), object()))
    monkeypatch.setattr(
        benchmark,
        "profile_operation",
        lambda operation, **_kwargs: captured.setdefault("operation", operation),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_pair",
        lambda *_args, **_kwargs: pytest.fail("profile mode entered benchmark timing"),
    )

    benchmark.main()

    assert captured["operation"].func is expected_operation


@pytest.mark.parametrize(
    "arguments",
    (
        ["--rows", "0", "--profile-path", "q8"],
        ["--rows", "1", "256", "--profile-path", "q8"],
        ["--rows", "1", "--profile-path", "q8", "--profile-iterations", "0"],
    ),
)
def test_main_rejects_invalid_profile_arguments(monkeypatch, arguments, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["q8_mps_linear.py", *arguments])

    with pytest.raises(SystemExit):
        benchmark.main()

    assert "error:" in capsys.readouterr().err
