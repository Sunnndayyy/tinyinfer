import json
from pathlib import Path

import pytest
import torch

from tinyinfer import benchmark, roofline

QWEN_SHAPES = roofline.QWEN_SHAPES
operation_counts = roofline.operation_counts


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


def test_correctness_check_compares_q8_with_the_bf16_reference() -> None:
    reference = torch.tensor([[1000.0, -32.0]], dtype=torch.bfloat16)

    roofline.check_correctness(reference.clone(), reference)

    with pytest.raises(AssertionError):
        roofline.check_correctness(reference + 128, reference)


def test_counter_bytes_are_measured_separately_from_model_bytes(tmp_path) -> None:
    path = tmp_path / "counters.json"
    path.write_text(
        json.dumps(
            {
                "shape": {"input_width": 32, "output_width": 64},
                "measurements": {
                    "q8-r1": {
                        "device_read_bytes": 13_000_000,
                        "device_write_bytes": 10_000,
                    }
                },
            }
        )
    )

    counters = roofline.load_hardware_counters(path, input_width=32, output_width=64)

    assert counters["q8-r1"].total_bytes == 13_010_000
    assert counters["q8-r1"].arithmetic_intensity(26_000_000) == pytest.approx(
        26_000_000 / 13_010_000
    )

    with pytest.raises(ValueError, match="counter shape"):
        roofline.load_hardware_counters(path, input_width=32, output_width=128)


def test_result_artifacts_label_model_and_counter_values(tmp_path) -> None:
    measurements = [
        roofline.Measurement(
            rows=1,
            weight_format="q8",
            milliseconds=0.25,
            counts=roofline.OperationCounts(flops=1_000_000, bytes=500_000),
        ),
        roofline.Measurement(
            rows=1,
            weight_format="bf16",
            milliseconds=0.5,
            counts=roofline.OperationCounts(flops=1_000_000, bytes=1_000_000),
        ),
        roofline.Measurement(
            rows=256,
            weight_format="q8",
            milliseconds=2.0,
            counts=roofline.OperationCounts(flops=256_000_000, bytes=1_000_000),
        ),
        roofline.Measurement(
            rows=256,
            weight_format="bf16",
            milliseconds=1.0,
            counts=roofline.OperationCounts(flops=256_000_000, bytes=2_000_000),
        ),
    ]
    counters = {
        "q8-r1": roofline.HardwareCounters(
            device_read_bytes=600_000,
            device_write_bytes=25_000,
        )
    }
    json_path = tmp_path / "results.json"
    plot_path = tmp_path / "roofline.svg"

    roofline.write_results_json(
        json_path,
        measurements,
        counters=counters,
        shape_name="test",
        input_width=32,
        output_width=64,
        warmup=2,
        repetitions=3,
        seed=17,
    )
    roofline.write_roofline_svg(
        plot_path,
        measurements,
        counters=counters,
        compute_reference_tflops=0.256,
    )

    payload = json.loads(json_path.read_text())
    assert payload["seed"] == 17
    assert payload["measurements"][0]["model_bytes"] == 500_000
    assert payload["measurements"][0]["counter_device_read_bytes"] == 600_000
    assert payload["measurements"][0]["counter_device_write_bytes"] == 25_000
    assert payload["measurements"][0]["counter_bytes"] == 625_000
    assert payload["measurements"][0]["counter_arithmetic_intensity"] == 1.6
    assert payload["measurements"][0]["counter_correlated_bandwidth_gb_s"] == 2.5
    svg = plot_path.read_text()
    assert "ideal algorithmic intensity" in svg
    assert "measured device-traffic intensity" in svg
    assert "advertised bandwidth, not measured" in svg
    assert "best BF16 point, not a ceiling" in svg
    assert svg.count('class="series"') == 2


def test_metal_capture_suite_warms_then_captures_one_operation_each(monkeypatch, tmp_path) -> None:
    events = []

    class Capture:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            events.append(f"begin:{self.label}")

        def __exit__(self, *_args):
            Path(f"0000-{self.label}.gputrace").mkdir()
            events.append(f"end:{self.label}")

    def operation(name):
        return lambda: events.append(name)

    original_directory = Path.cwd()
    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    monkeypatch.setattr("torch.mps.synchronize", lambda: events.append("sync"))
    monkeypatch.setattr("torch.mps.profiler.metal_capture", Capture)

    captures = roofline.metal_capture_suite(
        (("bf16-r1", operation("bf16")), ("q8-r1", operation("q8"))),
        output_dir=tmp_path,
        warmup=1,
    )

    assert Path.cwd() == original_directory
    assert events == [
        "bf16",
        "q8",
        "sync",
        "begin:bf16-r1",
        "bf16",
        "end:bf16-r1",
        "begin:q8-r1",
        "q8",
        "end:q8-r1",
    ]
    assert [path.name for path in captures] == [
        "0000-bf16-r1.gputrace",
        "0000-q8-r1.gputrace",
    ]


def test_metal_capture_suite_requires_capture_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)

    with pytest.raises(RuntimeError, match="MTL_CAPTURE_ENABLED=1"):
        roofline.metal_capture_suite((), output_dir=tmp_path, warmup=0)


def test_mps_pair_timing_alternates_and_synchronizes_samples(monkeypatch) -> None:
    events = []
    clock = iter((0.000, 0.001, 0.001, 0.003, 0.003, 0.007, 0.007, 0.012))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.torch.mps, "synchronize", lambda: events.append("sync"))

    q8_ms, bf16_ms = benchmark.measure_mps_pair(
        lambda: events.append("q8"),
        lambda: events.append("bf16"),
        warmup=1,
        repetitions=2,
    )

    assert events == [
        "q8",
        "bf16",
        "sync",
        "q8",
        "sync",
        "bf16",
        "sync",
        "bf16",
        "sync",
        "q8",
        "sync",
    ]
    assert q8_ms == pytest.approx(3.0)
    assert bf16_ms == pytest.approx(3.0)


def test_default_run_uses_fixed_local_artifact_paths(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(roofline, "main", lambda arguments: calls.append(arguments) or 0)

    assert roofline.run_default() == 0
    assert calls == [
        [
            "--json",
            ".tinyinfer/roofline/results.json",
            "--plot",
            ".tinyinfer/roofline/roofline.svg",
        ]
    ]


def test_default_run_uses_fixed_counter_file_when_present(tmp_path, monkeypatch) -> None:
    calls = []
    counter_file = tmp_path / ".tinyinfer" / "roofline" / "counters.json"
    counter_file.parent.mkdir(parents=True)
    counter_file.write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(roofline, "main", lambda arguments: calls.append(arguments) or 0)

    assert roofline.run_default() == 0
    assert calls[0][-2:] == ["--counters", ".tinyinfer/roofline/counters.json"]


def test_capture_run_replaces_only_stale_capture_packages(tmp_path, monkeypatch) -> None:
    calls = []
    capture_dir = tmp_path / ".tinyinfer" / "roofline" / "captures"
    capture_dir.mkdir(parents=True)
    (capture_dir / "stale.gputrace").mkdir()
    result_file = capture_dir.parent / "results.json"
    result_file.write_text("result")
    monkeypatch.chdir(tmp_path)

    def capture(arguments):
        calls.append(arguments)
        output_dir = Path(arguments[1])
        output_dir.mkdir(parents=True)
        (output_dir / "0000-new.gputrace").mkdir()
        return 0

    monkeypatch.setattr(roofline, "main", capture)

    assert roofline.run_default(capture=True) == 0
    assert not (capture_dir / "stale.gputrace").exists()
    assert (capture_dir / "0000-new.gputrace").exists()
    assert result_file.read_text() == "result"
    assert calls == [["--metal-capture-dir", ".tinyinfer/roofline/captures"]]


def test_failed_capture_keeps_the_last_complete_suite(tmp_path, monkeypatch) -> None:
    capture_dir = tmp_path / ".tinyinfer" / "roofline" / "captures"
    capture_dir.mkdir(parents=True)
    (capture_dir / "complete.gputrace").mkdir()
    monkeypatch.chdir(tmp_path)

    def fail_capture(arguments):
        staging_dir = Path(arguments[1])
        staging_dir.mkdir(parents=True)
        (staging_dir / "partial.gputrace").mkdir()
        return 1

    monkeypatch.setattr(roofline, "main", fail_capture)

    assert roofline.run_default(capture=True) == 1
    assert (capture_dir / "complete.gputrace").exists()
    assert not (capture_dir.parent / ".captures.previous").exists()


def test_main_routes_direct_capture_without_running_benchmark(monkeypatch, tmp_path) -> None:
    captured = {}

    def capture(operations, **_kwargs):
        captured["operations"] = operations
        return []

    monkeypatch.setenv("MTL_CAPTURE_ENABLED", "1")
    monkeypatch.setattr(roofline, "require_q8_mps", lambda: None)
    monkeypatch.setattr(roofline, "hardware_name", lambda: "test GPU")
    monkeypatch.setattr(roofline.torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(
        roofline.torch,
        "randn",
        lambda shape, **_kwargs: torch.empty(shape),
    )
    monkeypatch.setattr(roofline, "make_weights", lambda *_widths: (object(), object(), object()))
    monkeypatch.setattr(
        roofline,
        "metal_capture_suite",
        capture,
    )
    monkeypatch.setattr(
        roofline,
        "measure_mps_pair",
        lambda *_args, **_kwargs: pytest.fail("profile mode entered benchmark timing"),
    )

    roofline.main(["--metal-capture-dir", str(tmp_path)])

    assert [label for label, _operation in captured["operations"]] == [
        "bf16-r1",
        "q8-r1",
        "bf16-r256",
        "q8-r256",
    ]
    assert [operation.func for _label, operation in captured["operations"]] == [
        roofline.F.linear,
        roofline._q8_linear_mps,
        roofline.F.linear,
        roofline._q8_linear_mps,
    ]


@pytest.mark.parametrize(
    "arguments",
    (
        ["--rows", "0"],
        ["--compute-reference-tflops", "0"],
        ["--rows", "1", "--plot", "/tmp/plot.svg"],
    ),
)
def test_main_rejects_invalid_profile_arguments(arguments, capsys) -> None:
    with pytest.raises(SystemExit):
        roofline.main(arguments)

    assert "error:" in capsys.readouterr().err
