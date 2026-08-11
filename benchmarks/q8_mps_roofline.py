"""Teach the decode-to-prefill Roofline transition with one Qwen linear layer."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from contextlib import chdir
from dataclasses import dataclass
from functools import partial
from html import escape
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from tinyinfer.benchmark import hardware_name
from tinyinfer.quantization.int8 import GROUP_SIZE, Q8_MAX
from tinyinfer.quantization.metal import _q8_linear_mps, require_q8_mps

OperationPath = Literal["bf16", "q8"]

# Qwen2.5-1.5B-Instruct uses hidden_size=1536, intermediate_size=8960,
# and vocab_size=151936. The MLP projection is small enough for a useful row sweep.
QWEN_SHAPES = {
    "mlp-up": (1536, 8960),
    "lm-head": (1536, 151936),
}
DEFAULT_ROWS = (1, 4, 16, 64, 256)
CAPTURE_ROWS = (1, 256)
M4_PRO_ADVERTISED_BANDWIDTH_GB_S = 273.0


@dataclass(frozen=True)
class OperationCounts:
    flops: int
    bytes: int

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes


@dataclass(frozen=True)
class Measurement:
    rows: int
    weight_format: OperationPath
    milliseconds: float
    counts: OperationCounts

    @property
    def achieved_tflops(self) -> float:
        return self.counts.flops / (self.milliseconds * 1e9)

    @property
    def model_bandwidth_gb_s(self) -> float:
        return self.counts.bytes / (self.milliseconds * 1e6)


@dataclass(frozen=True)
class HardwareCounters:
    """Device traffic copied from an Xcode GPU capture."""

    device_read_bytes: int
    device_write_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.device_read_bytes + self.device_write_bytes

    def arithmetic_intensity(self, flops: int) -> float:
        return flops / self.total_bytes


def result_key(weight_format: OperationPath, rows: int) -> str:
    return f"{weight_format}-r{rows}"


def check_correctness(q8_output: Tensor, bf16_output: Tensor) -> None:
    """Check the fused output against the matched BF16 operation."""
    torch.testing.assert_close(q8_output, bf16_output, rtol=0.02, atol=1.0)


def load_hardware_counters(
    path: Path,
    *,
    input_width: int,
    output_width: int,
) -> dict[str, HardwareCounters]:
    """Load device-byte counters copied from Xcode's capture viewer."""
    payload = json.loads(path.read_text())
    expected_shape = {"input_width": input_width, "output_width": output_width}
    if payload.get("shape") != expected_shape:
        raise ValueError(f"counter shape must be {expected_shape}")
    counters = {}
    for key, values in payload["measurements"].items():
        read_bytes = int(values["device_read_bytes"])
        write_bytes = int(values["device_write_bytes"])
        if read_bytes < 0 or write_bytes < 0 or read_bytes + write_bytes == 0:
            raise ValueError(f"{key}: counter bytes must have a positive total")
        counters[key] = HardwareCounters(read_bytes, write_bytes)
    return counters


def _measurement_dict(
    measurement: Measurement,
    counters: dict[str, HardwareCounters],
) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": measurement.rows,
        "path": measurement.weight_format,
        "milliseconds": measurement.milliseconds,
        "flops": measurement.counts.flops,
        "model_bytes": measurement.counts.bytes,
        "model_arithmetic_intensity": measurement.counts.arithmetic_intensity,
        "achieved_tflops": measurement.achieved_tflops,
        "model_bandwidth_gb_s": measurement.model_bandwidth_gb_s,
    }
    counter = counters.get(result_key(measurement.weight_format, measurement.rows))
    if counter is not None:
        result["counter_device_read_bytes"] = counter.device_read_bytes
        result["counter_device_write_bytes"] = counter.device_write_bytes
        result["counter_bytes"] = counter.total_bytes
        result["counter_arithmetic_intensity"] = counter.arithmetic_intensity(
            measurement.counts.flops
        )
        result["counter_correlated_bandwidth_gb_s"] = counter.total_bytes / (
            measurement.milliseconds * 1e6
        )
    return result


def write_results_json(
    path: Path,
    measurements: list[Measurement],
    *,
    counters: dict[str, HardwareCounters],
    shape_name: str,
    input_width: int,
    output_width: int,
    warmup: int,
    repetitions: int,
    seed: int,
) -> None:
    payload = {
        "hardware": hardware_name(),
        "pytorch": torch.__version__,
        "shape": {
            "name": shape_name,
            "input_width": input_width,
            "output_width": output_width,
        },
        "warmup": warmup,
        "repetitions": repetitions,
        "seed": seed,
        "measurement_notes": {
            "runtime": "median synchronized wall time",
            "model_bytes": "ideal algorithmic bytes; not hardware traffic",
            "counter_bytes": "Xcode device read plus device write bytes",
        },
        "measurements": [_measurement_dict(measurement, counters) for measurement in measurements],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_roofline_svg(
    path: Path,
    measurements: list[Measurement],
    *,
    counters: dict[str, HardwareCounters],
    compute_reference_tflops: float,
) -> None:
    """Write one dependency-free teaching Roofline graph."""
    width, height = 920, 600
    left, top, right, bottom = 92, 88, 32, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    model_x = [result.counts.arithmetic_intensity for result in measurements]
    counter_x = [
        counter.arithmetic_intensity(result.counts.flops)
        for result in measurements
        if (counter := counters.get(result_key(result.weight_format, result.rows)))
    ]
    achieved_y = [result.achieved_tflops for result in measurements]
    all_x = model_x + counter_x
    x_min = min(0.5, min(all_x) / 1.6)
    x_max = max(all_x) * 1.6
    y_min = max(min(achieved_y) / 2, 0.001)
    y_max = max(compute_reference_tflops, max(achieved_y)) * 1.8

    def x_position(value: float) -> float:
        return left + plot_width * (
            (math.log10(value) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min))
        )

    def y_position(value: float) -> float:
        return top + plot_height * (
            1 - (math.log10(value) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min))
        )

    def decade_ticks(low: float, high: float) -> list[float]:
        start = math.floor(math.log10(low))
        stop = math.ceil(math.log10(high))
        return [10.0**power for power in range(start, stop + 1) if low <= 10.0**power <= high]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,sans-serif;fill:#202124}.grid{stroke:#d9dde3;stroke-width:1}.axis{stroke:#59636e;stroke-width:1.5}.ref{fill:none;stroke:#777;stroke-width:2}.label{font-size:13px}.small{font-size:12px}</style>",
        f'<text x="{left}" y="28" font-size="20" font-weight="600">TinyInfer Q8/BF16 teaching Roofline</text>',
        '<circle cx="100" cy="56" r="5" fill="#555"/><text class="small" x="112" y="60">ideal algorithmic intensity</text>',
        '<rect x="286" y="51" width="10" height="10" fill="white" stroke="#555" stroke-width="2"/><text class="small" x="304" y="60">measured device-traffic intensity</text>',
        '<line x1="527" y1="56" x2="553" y2="56" class="ref"/><text class="small" x="561" y="52">advertised bandwidth, not measured</text>',
        '<text class="small" x="561" y="68">best BF16 point, not a ceiling</text>',
    ]
    for tick in decade_ticks(x_min, x_max):
        x = x_position(tick)
        lines.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>',
                f'<text class="small" x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle">{tick:g}</text>',
            ]
        )
    for tick in decade_ticks(y_min, y_max):
        y = y_position(tick)
        lines.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>',
                f'<text class="small" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>',
            ]
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text class="label" x="{left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle">Arithmetic intensity (FLOP/byte, log scale)</text>',
            f'<text class="label" x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">Achieved TFLOP/s (log scale)</text>',
        ]
    )
    reference_segments = 80
    reference_points = []
    for index in range(reference_segments + 1):
        intensity = 10 ** (
            math.log10(x_min) + index * (math.log10(x_max) - math.log10(x_min)) / reference_segments
        )
        performance = min(
            compute_reference_tflops,
            M4_PRO_ADVERTISED_BANDWIDTH_GB_S * intensity / 1000,
        )
        reference_points.append(f"{x_position(intensity):.1f},{y_position(performance):.1f}")
    lines.append(f'<polyline class="ref" points="{" ".join(reference_points)}"/>')
    colors = {"bf16": "#2563eb", "q8": "#dc2626"}
    for result in measurements:
        color = colors[result.weight_format]
        x = x_position(result.counts.arithmetic_intensity)
        y = y_position(result.achieved_tflops)
        label = escape(f"{result.weight_format.upper()} r{result.rows}")
        lines.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>',
                f'<text class="small" x="{x + 9:.1f}" y="{y - 7:.1f}">{label}</text>',
            ]
        )
        counter = counters.get(result_key(result.weight_format, result.rows))
        if counter is not None:
            counter_x_position = x_position(counter.arithmetic_intensity(result.counts.flops))
            lines.append(
                f'<rect x="{counter_x_position - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="white" stroke="{color}" stroke-width="2"/>'
            )
    lines.extend(
        [
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def operation_counts(
    weight_format: OperationPath,
    *,
    rows: int,
    input_width: int,
    output_width: int,
) -> OperationCounts:
    """Return matmul FLOPs and ideal algorithmic bytes, not hardware counters."""
    flops = 2 * rows * input_width * output_width
    activation_bytes = torch.bfloat16.itemsize * rows * input_width
    output_bytes = torch.bfloat16.itemsize * rows * output_width
    if weight_format == "bf16":
        weight_bytes = torch.bfloat16.itemsize * input_width * output_width
    else:
        weight_bytes = torch.int8.itemsize * input_width * output_width
        weight_bytes += torch.float16.itemsize * input_width * output_width // GROUP_SIZE
    return OperationCounts(flops, activation_bytes + weight_bytes + output_bytes)


def sample_order(sample_index: int) -> tuple[str, str]:
    return ("q8", "bf16") if sample_index % 2 == 0 else ("bf16", "q8")


def measure_pair(
    q8_operation: Callable[[], Tensor],
    bf16_operation: Callable[[], Tensor],
    *,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    operations = {"q8": q8_operation, "bf16": bf16_operation}
    for sample_index in range(warmup):
        for name in sample_order(sample_index):
            operations[name]()

    # Finish shader compilation and lazy MPS work before synchronized samples.
    torch.mps.synchronize()
    samples = {"q8": [], "bf16": []}
    for sample_index in range(repetitions):
        for name in sample_order(sample_index):
            started = time.perf_counter()
            operations[name]()
            torch.mps.synchronize()
            samples[name].append((time.perf_counter() - started) * 1000)
    return statistics.median(samples["q8"]), statistics.median(samples["bf16"])


def metal_capture_suite(
    operations: tuple[tuple[str, Callable[[], Tensor]], ...],
    *,
    output_dir: Path,
    warmup: int,
) -> list[Path]:
    """Write one direct Xcode GPU capture for each single operation."""
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("run with MTL_CAPTURE_ENABLED=1 to enable Metal capture")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, operation in operations:
        for _ in range(warmup):
            operation()
    torch.mps.synchronize()

    captures = []
    with chdir(output_dir):
        for label, operation in operations:
            before = set(Path.cwd().glob(f"*-{label}.gputrace"))
            with torch.mps.profiler.metal_capture(label):
                operation()
            created = sorted(set(Path.cwd().glob(f"*-{label}.gputrace")) - before)
            if len(created) != 1:
                raise RuntimeError(f"expected one new capture for {label}, found {len(created)}")
            captures.append(created[0].resolve())
    return captures


def make_weights(input_width: int, output_width: int) -> tuple[Tensor, Tensor, Tensor]:
    q8_weights = torch.randint(
        -Q8_MAX,
        Q8_MAX + 1,
        (output_width, input_width),
        dtype=torch.int8,
        device="mps",
    )
    scales = torch.ones(
        (output_width, input_width // GROUP_SIZE),
        dtype=torch.float16,
        device="mps",
    )
    bf16_weights = q8_weights.to(torch.bfloat16)
    return q8_weights, scales, bf16_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=QWEN_SHAPES, default="mlp-up")
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-reference-tflops", type=float)
    parser.add_argument(
        "--metal-capture-dir",
        type=Path,
        help=f"write {len(CAPTURE_ROWS) * 2} direct .gputrace captures",
    )
    parser.add_argument("--counters", type=Path, help="Xcode device-byte counter JSON")
    parser.add_argument("--json", type=Path, help="write benchmark results as JSON")
    parser.add_argument("--plot", type=Path, help="write a Roofline SVG")
    args = parser.parse_args()

    if any(rows < 1 for rows in args.rows):
        parser.error("--rows values must be positive")
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("--warmup must be nonnegative and --repetitions must be positive")
    if args.metal_capture_dir is not None and any((args.counters, args.json, args.plot)):
        parser.error("counter, JSON, and plot outputs are benchmark-only options")
    if args.compute_reference_tflops is not None and args.compute_reference_tflops <= 0:
        parser.error("--compute-reference-tflops must be positive")
    if args.metal_capture_dir is not None and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        parser.error("--metal-capture-dir requires MTL_CAPTURE_ENABLED=1")
    if args.plot and args.compute_reference_tflops is None and max(args.rows) < 64:
        parser.error("--plot needs at least 64 rows or --compute-reference-tflops")

    require_q8_mps()
    torch.manual_seed(args.seed)
    input_width, output_width = QWEN_SHAPES[args.shape]
    max_rows = max(CAPTURE_ROWS) if args.metal_capture_dir else max(args.rows)
    inputs = torch.randn((max_rows, input_width), dtype=torch.bfloat16, device="mps")
    if args.metal_capture_dir is not None:
        q8_weights, scales, bf16_weights = make_weights(input_width, output_width)
        operations = tuple(
            operation
            for rows in CAPTURE_ROWS
            for operation in (
                (result_key("bf16", rows), partial(F.linear, inputs[:rows], bf16_weights)),
                (
                    result_key("q8", rows),
                    partial(_q8_linear_mps, inputs[:rows], q8_weights, scales),
                ),
            )
        )
        print(f"hardware: {hardware_name()}")
        print(f"shape: [rows, {input_width}] x [{output_width}, {input_width}]")
        print(f"capture suite: bf16/q8 at rows {CAPTURE_ROWS}")
        print("each capture contains one operation; capture timing is not benchmark timing")
        captures = metal_capture_suite(
            operations,
            output_dir=args.metal_capture_dir,
            warmup=args.warmup,
        )
        for capture in captures:
            print(f"  {capture}")
        return
    counters = (
        load_hardware_counters(
            args.counters,
            input_width=input_width,
            output_width=output_width,
        )
        if args.counters
        else {}
    )
    q8_weights, scales, bf16_weights = make_weights(input_width, output_width)
    measurements = []
    for rows in args.rows:
        row_inputs = inputs[:rows]
        q8_operation = partial(_q8_linear_mps, row_inputs, q8_weights, scales)
        bf16_operation = partial(F.linear, row_inputs, bf16_weights)
        check_correctness(q8_operation(), bf16_operation())
        q8_ms, bf16_ms = measure_pair(
            q8_operation,
            bf16_operation,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        for weight_format, milliseconds in (("bf16", bf16_ms), ("q8", q8_ms)):
            counts = operation_counts(
                weight_format,
                rows=rows,
                input_width=input_width,
                output_width=output_width,
            )
            measurements.append(Measurement(rows, weight_format, milliseconds, counts))

    measured_compute_reference = max(
        result.achieved_tflops for result in measurements if result.weight_format == "bf16"
    )
    compute_reference = args.compute_reference_tflops
    compute_reference_label = "supplied estimate"
    if compute_reference is None and max(args.rows) >= 64:
        compute_reference = measured_compute_reference
        compute_reference_label = "best BF16 point in this sweep"
    ridge_intensity = (
        compute_reference * 1000 / M4_PRO_ADVERTISED_BANDWIDTH_GB_S
        if compute_reference is not None
        else None
    )

    print(f"hardware: {hardware_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"shape: [{input_width}] x [{output_width}, {input_width}] ({args.shape})")
    print(
        f"rows: {args.rows}; warmup: {args.warmup}; "
        f"repetitions: {args.repetitions}; seed: {args.seed}"
    )
    print("correctness: Q8 matches BF16 (rtol=0.02, atol=1.0)")
    print()
    print("rows  path     ms   FLOP/byte   TFLOP/s   model GB/s   model side")
    for result in measurements:
        if ridge_intensity is None:
            side = "n/a"
        else:
            side = "memory" if result.counts.arithmetic_intensity < ridge_intensity else "compute"
        print(
            f"{result.rows:>4}  {result.weight_format:<4}  {result.milliseconds:>7.3f}"
            f"  {result.counts.arithmetic_intensity:>10.2f}"
            f"  {result.achieved_tflops:>8.3f}"
            f"  {result.model_bandwidth_gb_s:>11.1f}   {side}"
        )

    print()
    print("Assumptions (ideal algorithmic model, not MPS/Metal hardware counters):")
    print("  FLOPs = 2 * rows * input_width * output_width (matmul only).")
    print("  BF16 bytes = input + BF16 weights + output, each counted once.")
    print("  Q8 bytes = input + INT8 weights + FP16 scale/32 weights + output, once.")
    print("  model GB/s = those ideal bytes / synchronized runtime; it is not measured traffic.")
    print("  The row-wise Q8 shader does not explicitly reuse weights between rows.")
    print()
    print("Teaching Roofline references (not measured sustainable ceilings):")
    print(
        f"  bandwidth = {M4_PRO_ADVERTISED_BANDWIDTH_GB_S:.0f} GB/s "
        "Apple advertised unified-memory bandwidth."
    )
    if compute_reference is None:
        print("  compute/crossover = n/a; include at least 64 rows or supply a reference.")
    else:
        print(
            f"  compute = {compute_reference:.3f} TFLOP/s, {compute_reference_label} "
            "(not a peak ceiling)."
        )
        print(f"  estimated crossover = {ridge_intensity:.2f} FLOP/byte.")
    print("  'model side' is therefore a teaching-model classification, not counter evidence.")
    if counters:
        print()
        print("Xcode counter evidence (copied from separate capture replay):")
        for result in measurements:
            key = result_key(result.weight_format, result.rows)
            if counter := counters.get(key):
                print(
                    f"  {key}: {counter.total_bytes / 1e6:.2f} MB device traffic, "
                    f"{counter.arithmetic_intensity(result.counts.flops):.2f} FLOP/byte"
                )

    if args.json:
        write_results_json(
            args.json,
            measurements,
            counters=counters,
            shape_name=args.shape,
            input_width=input_width,
            output_width=output_width,
            warmup=args.warmup,
            repetitions=args.repetitions,
            seed=args.seed,
        )
        print(f"results: {args.json.resolve()}")
    if args.plot:
        write_roofline_svg(
            args.plot,
            measurements,
            counters=counters,
            compute_reference_tflops=compute_reference,
        )
        print(f"plot: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
