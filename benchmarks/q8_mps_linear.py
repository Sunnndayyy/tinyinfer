"""Compare TinyInfer's fused Q8 Metal linear operation with MPS BF16."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from tinyinfer.benchmark import hardware_name
from tinyinfer.quantization.int8 import GROUP_SIZE
from tinyinfer.quantization.metal import _q8_linear_mps, require_q8_mps

SHAPES = ((1, 1536, 8960), (1, 1536, 151936))


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

    # Finish shader compilation and lazy MPS work before timing synchronized samples.
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


def benchmark_shape(
    rows: int,
    input_width: int,
    output_width: int,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    inputs = torch.randn((rows, input_width), dtype=torch.bfloat16, device="mps")
    q8_weights = torch.randint(
        -127,
        128,
        (output_width, input_width),
        dtype=torch.int8,
    ).to("mps")
    scales = torch.ones(
        (output_width, input_width // GROUP_SIZE),
        dtype=torch.float16,
        device="mps",
    )
    bf16_weights = q8_weights.to(torch.bfloat16)

    return measure_pair(
        lambda: _q8_linear_mps(inputs, q8_weights, scales),
        lambda: F.linear(inputs, bf16_weights),
        warmup=warmup,
        repetitions=repetitions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    require_q8_mps()
    torch.manual_seed(args.seed)

    print(f"hardware: {hardware_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"warmup: {args.warmup}; repetitions: {args.repetitions}; seed: {args.seed}")
    print("shape                                   Q8 ms    BF16 ms    speedup")
    for rows, input_width, output_width in SHAPES:
        q8_ms, bf16_ms = benchmark_shape(
            rows,
            input_width,
            output_width,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        shape = f"[{rows}, {input_width}] x [{output_width}, {input_width}]"
        print(f"{shape:<38} {q8_ms:>7.3f}    {bf16_ms:>7.3f}    {bf16_ms / q8_ms:>6.3f}x")
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
