"""Readable Q8 reference operations."""

import torch
from torch import Tensor

GROUP_SIZE = 32
Q8_MAX = 127


def _quantize_q8_groups(groups: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize tensors whose final dimension is one complete Q8 group."""
    working = groups.to(torch.float32)
    if not torch.isfinite(working).all():
        raise ValueError("Input weights must be finite after conversion to float32")

    largest_abs = working.abs().amax(dim=-1, keepdim=True)
    scales = largest_abs / Q8_MAX
    computed_scales = scales.squeeze(-1)
    stored_scales = computed_scales.to(torch.float16)
    stored_scales_are_valid = torch.isfinite(stored_scales) & (
        (computed_scales == 0) | (stored_scales > 0)
    )
    if not stored_scales_are_valid.all():
        raise ValueError("Q8 scales must be representable as non-zero finite float16 values")

    # Zero groups need a safe divisor, but keep their stored scale equal to zero.
    divisors = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = working.div(divisors)
    quantized.round_().clamp_(-Q8_MAX, Q8_MAX)
    return quantized.to(torch.int8), stored_scales


def quantize_q8_group(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize one group of 32 floating-point weights."""
    if tensor.ndim != 1 or tensor.shape[0] != GROUP_SIZE:
        raise ValueError("Input must be one-dimensional with exactly 32 weights")
    if not tensor.is_floating_point():
        raise ValueError("Input weights must use a floating-point dtype")

    return _quantize_q8_groups(tensor)


def quantize_q8(weights: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a two-dimensional weight matrix in groups of 32 columns."""
    if weights.ndim != 2:
        raise ValueError("Weights must be a two-dimensional matrix")
    if weights.shape[1] == 0 or weights.shape[1] % GROUP_SIZE != 0:
        raise ValueError("Weight input width must be divisible by 32")
    if not weights.is_floating_point():
        raise ValueError("Input weights must use a floating-point dtype")
    rows, columns = weights.shape
    groups = weights.reshape(rows, columns // GROUP_SIZE, GROUP_SIZE)
    quantized, scales = _quantize_q8_groups(groups)
    return quantized.reshape_as(weights), scales


def dequantize_q8_group(quantized: Tensor, scale: Tensor) -> Tensor:
    """Approximately restore a group of Q8 weights."""
    if quantized.shape != (GROUP_SIZE,):
        raise ValueError("Quantized weights must have shape (32,)")
    if scale.shape != ():
        raise ValueError("Scale must be scalar")
    return quantized.to(torch.float32) * scale.to(torch.float32)


def dequantize_q8(quantized: Tensor, scales: Tensor) -> Tensor:
    """Restore a full Q8 weight matrix as FP32 for reference calculations."""
    if quantized.ndim != 2 or quantized.shape[1] == 0:
        raise ValueError("Quantized weights must be a non-empty two-dimensional matrix")
    rows, columns = quantized.shape
    if columns % GROUP_SIZE != 0:
        raise ValueError("Quantized weight input width must be divisible by 32")
    groups_per_row = columns // GROUP_SIZE
    if scales.shape != (rows, groups_per_row):
        raise ValueError("Scales must have shape (rows, columns / 32)")

    groups = quantized.reshape(rows, groups_per_row, GROUP_SIZE)
    restored = groups.to(torch.float32)
    # Shape (rows, groups, 1) broadcasts one scale over each 32-value group.
    restored.mul_(scales.to(torch.float32).unsqueeze(-1))
    return restored.reshape(rows, columns)
