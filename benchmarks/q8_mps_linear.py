"""Compare TinyInfer's fused Q8 Metal linear operation with MPS BF16."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tinyinfer.benchmark import hardware_name, measure_mps_pair
from tinyinfer.quantization.int8 import GROUP_SIZE
from tinyinfer.quantization.metal import _q8_linear_mps, require_q8_mps

ROWS = (1, 8, 32, 128, 512)


@dataclass(frozen=True)
class LinearShape:
    name: str
    input_width: int
    output_width: int
    rows: tuple[int, ...] = ROWS


# Every distinct linear shape in Qwen2.5-1.5B. TinyInfer projects only the final
# hidden state to vocabulary logits, so the tied embedding stays a one-row case.
SHAPES = (
    LinearShape("attention k/v", 1536, 256),
    LinearShape("attention q/o", 1536, 1536),
    LinearShape("MLP gate/up", 1536, 8960),
    LinearShape("MLP down", 8960, 1536),
    LinearShape("tied vocabulary", 1536, 151936, (1,)),
)


def benchmark_shape(
    rows: int,
    q8_weights: Tensor,
    scales: Tensor,
    bf16_weights: Tensor,
    *,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    input_width = q8_weights.shape[1]
    inputs = torch.randn((rows, input_width), dtype=torch.bfloat16, device="mps")

    return measure_mps_pair(
        lambda: _q8_linear_mps(inputs, q8_weights, scales),
        lambda: F.linear(inputs, bf16_weights),
        warmup=warmup,
        repetitions=repetitions,
    )


def create_weights(input_width: int, output_width: int) -> tuple[Tensor, Tensor, Tensor]:
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
    return q8_weights, scales, bf16_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    require_q8_mps()
    torch.manual_seed(args.seed)

    print(f"hardware: {hardware_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"warmup: {args.warmup}; repetitions: {args.repetitions}; seed: {args.seed}")
    print("operation        activation x weight                  Q8 ms    BF16 ms   BF16/Q8")
    for linear_shape in SHAPES:
        weights = create_weights(linear_shape.input_width, linear_shape.output_width)
        for rows in linear_shape.rows:
            q8_ms, bf16_ms = benchmark_shape(
                rows,
                *weights,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            shape = (
                f"[{rows}, {linear_shape.input_width}] x "
                f"[{linear_shape.output_width}, {linear_shape.input_width}]"
            )
            print(
                f"{linear_shape.name:<16} {shape:<36} "
                f"{q8_ms:>7.3f}    {bf16_ms:>7.3f}    {bf16_ms / q8_ms:>6.3f}x"
            )
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
