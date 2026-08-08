"""Teach the decode-to-prefill Roofline transition with one Qwen linear layer."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
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


def measure_pair(
    q8_operation: Callable[[], Tensor],
    bf16_operation: Callable[[], Tensor],
    *,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        q8_operation()
        bf16_operation()

    # Finish shader compilation and lazy MPS work before synchronized samples.
    torch.mps.synchronize()
    q8_samples = []
    bf16_samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        q8_operation()
        torch.mps.synchronize()
        q8_samples.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        bf16_operation()
        torch.mps.synchronize()
        bf16_samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(q8_samples), statistics.median(bf16_samples)


def profile_operation(
    operation: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
) -> None:
    """Emit MPS signposts for one operation; this is not benchmark timing."""
    for _ in range(warmup):
        operation()
    torch.mps.synchronize()

    with torch.mps.profiler.profile("interval,event", wait_until_completed=True):
        for _ in range(iterations):
            operation()
    torch.mps.synchronize()


def make_q8_weights(input_width: int, output_width: int) -> tuple[Tensor, Tensor]:
    q8_weights = torch.randint(
        -Q8_MAX,
        Q8_MAX + 1,
        (output_width, input_width),
        dtype=torch.int8,
    ).to("mps")
    scales = torch.ones(
        (output_width, input_width // GROUP_SIZE),
        dtype=torch.float16,
        device="mps",
    )
    return q8_weights, scales


def make_weights(input_width: int, output_width: int) -> tuple[Tensor, Tensor, Tensor]:
    q8_weights, scales = make_q8_weights(input_width, output_width)
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
    parser.add_argument("--profile-path", choices=("bf16", "q8"))
    parser.add_argument("--profile-iterations", type=int, default=20)
    args = parser.parse_args()

    if any(rows < 1 for rows in args.rows):
        parser.error("--rows values must be positive")
    if args.profile_path is not None and len(args.rows) != 1:
        parser.error("--profile-path requires exactly one --rows value")
    if args.profile_path is not None and args.profile_iterations < 1:
        parser.error("--profile-iterations must be positive")

    require_q8_mps()
    torch.manual_seed(args.seed)
    input_width, output_width = QWEN_SHAPES[args.shape]
    inputs = torch.randn((max(args.rows), input_width), dtype=torch.bfloat16, device="mps")
    if args.profile_path is not None:
        rows = args.rows[0]
        row_inputs = inputs[:rows]
        if args.profile_path == "q8":
            q8_weights, scales = make_q8_weights(input_width, output_width)
            operation = partial(_q8_linear_mps, row_inputs, q8_weights, scales)
        else:
            bf16_weights = torch.randn(
                (output_width, input_width), dtype=torch.bfloat16, device="mps"
            )
            operation = partial(F.linear, row_inputs, bf16_weights)
        print(f"hardware: {hardware_name()}")
        print(f"shape: [{rows}, {input_width}] x [{output_width}, {input_width}]")
        print(
            f"capture: {args.profile_path}; warmup: {args.warmup}; "
            f"iterations: {args.profile_iterations}"
        )
        print("capture mode emits MPS signposts and is not benchmark timing")
        profile_operation(
            operation,
            warmup=args.warmup,
            iterations=args.profile_iterations,
        )
        return

    q8_weights, scales, bf16_weights = make_weights(input_width, output_width)
    measurements = []
    for rows in args.rows:
        row_inputs = inputs[:rows]
        q8_ms, bf16_ms = measure_pair(
            partial(_q8_linear_mps, row_inputs, q8_weights, scales),
            partial(F.linear, row_inputs, bf16_weights),
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
    print(f"rows: {args.rows}; warmup: {args.warmup}; repetitions: {args.repetitions}")
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


if __name__ == "__main__":
    main()
