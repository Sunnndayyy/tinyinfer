"""Fused MPS operations for packed weights."""

from functools import lru_cache
from importlib import resources

import torch
from torch import Tensor

from .int8 import GROUP_SIZE

OUTPUTS_PER_THREADGROUP = 4
THREADGROUP_SIZE = 128
EMBEDDING_THREADGROUP_SIZE = 256


def q8_mps_available() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def require_q8_mps() -> None:
    if not q8_mps_available():
        raise RuntimeError("Q8 on MPS requires an available device and torch.mps.compile_shader")


@lru_cache(maxsize=1)
def _shader_library():
    source = resources.files("tinyinfer.quantization").joinpath("weight_only.metal").read_text()
    return torch.mps.compile_shader(source)


def q8_linear_mps(
    inputs: Tensor,
    weights: Tensor,
    scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Multiply BF16 activation rows by Q8 weights without restoring the weight matrix."""
    require_q8_mps()
    if inputs.device.type != "mps" or weights.device.type != "mps" or scales.device.type != "mps":
        raise ValueError("Q8 Metal inputs, weights, and scales must be on MPS")
    if inputs.dtype != torch.bfloat16:
        raise ValueError("Q8 Metal inputs must use bfloat16")
    if weights.dtype != torch.int8:
        raise ValueError("Q8 Metal weights must use int8")
    if scales.dtype != torch.float16:
        raise ValueError("Q8 Metal scales must use float16")
    if inputs.ndim == 0 or weights.ndim != 2 or inputs.shape[-1] != weights.shape[1]:
        raise ValueError("Input last dimension must match the quantized weight width")

    output_width, input_width = weights.shape
    if input_width == 0 or input_width % GROUP_SIZE:
        raise ValueError(f"Q8 Metal weight width must be divisible by {GROUP_SIZE}")
    if scales.shape != (output_width, input_width // GROUP_SIZE):
        raise ValueError("Q8 Metal scales have the wrong shape")
    if bias is not None and (
        bias.device.type != "mps" or bias.dtype != torch.bfloat16 or bias.shape != (output_width,)
    ):
        raise ValueError("Q8 Metal bias must be an MPS bfloat16 output vector")
    if not weights.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Q8 Metal weights and scales must be contiguous")

    return _q8_linear_mps(inputs, weights, scales, bias)


def _q8_linear_mps(
    inputs: Tensor,
    weights: Tensor,
    scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Dispatch tensors already validated while loading a Q8 model."""
    output_width, input_width = weights.shape

    inputs = inputs.contiguous()
    rows = inputs.numel() // input_width
    output = torch.empty((rows, output_width), device="mps", dtype=inputs.dtype)
    output_blocks = (output_width + OUTPUTS_PER_THREADGROUP - 1) // OUTPUTS_PER_THREADGROUP
    _shader_library().q8_linear_bf16(
        inputs,
        weights,
        scales,
        output if bias is None else bias,
        output,
        input_width,
        output_width,
        int(bias is not None),
        threads=(output_blocks * THREADGROUP_SIZE, rows, 1),
        group_size=(THREADGROUP_SIZE, 1, 1),
    )
    return output.reshape(*inputs.shape[:-1], output_width)


def q8_embedding_mps(input_ids: Tensor, weights: Tensor, scales: Tensor) -> Tensor:
    """Restore only the packed rows selected by MPS token IDs."""
    require_q8_mps()
    if (
        input_ids.device.type != "mps"
        or weights.device.type != "mps"
        or scales.device.type != "mps"
    ):
        raise ValueError("Q8 Metal token IDs, weights, and scales must be on MPS")
    if input_ids.dtype != torch.int64:
        raise ValueError("Q8 Metal token IDs must use int64")
    if weights.dtype != torch.int8 or weights.ndim != 2:
        raise ValueError("Q8 Metal embedding weights must be a two-dimensional int8 matrix")
    if scales.dtype != torch.float16:
        raise ValueError("Q8 Metal embedding scales must use float16")

    vocabulary_size, width = weights.shape
    if scales.shape != (vocabulary_size, width // GROUP_SIZE):
        raise ValueError("Q8 Metal embedding scales have the wrong shape")

    input_ids = input_ids.contiguous()
    output = torch.empty((*input_ids.shape, width), device="mps", dtype=torch.bfloat16)
    vectors = input_ids.numel() * width // 4
    _shader_library().q8_embedding_bf16(
        input_ids,
        weights,
        scales,
        output,
        width,
        threads=vectors,
        group_size=min(EMBEDDING_THREADGROUP_SIZE, vectors),
    )
    return output
