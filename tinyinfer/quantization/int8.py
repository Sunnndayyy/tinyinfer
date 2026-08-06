"""Readable Q8 reference operations."""

import torch
from torch import Tensor

GROUP_SIZE = 32
Q8_MAX = 127


def quantize_q8_group(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize one group of 32 floating-point weights."""
    if tensor.ndim != 1 or tensor.shape[0] != GROUP_SIZE:
        raise ValueError("Input must be one-dimensional with exactly 32 weights")
    if not tensor.is_floating_point():
        raise ValueError("Input weights must use a floating-point dtype")

    working = tensor.to(torch.float32)
    if not torch.isfinite(working).all():
        raise ValueError("Input weights must be finite")

    largest_abs = working.abs().max()

    # A zero scale is safe because every stored integer is also zero.
    if largest_abs == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), tensor.new_zeros((), dtype=torch.float16)

    # Calculate in FP32, then store the one scale in FP16 with the packed weights.
    scale = largest_abs / Q8_MAX
    quantized = (working / scale).round().clamp(-Q8_MAX, Q8_MAX).to(torch.int8)

    return quantized, scale.to(torch.float16)


def dequantize_q8_group(quantized: Tensor, scale: Tensor) -> Tensor:
    """Approximately restore a group of Q8 weights."""
    if quantized.dtype != torch.int8:
        raise ValueError("Quantized weights must use dtype int8")
    if quantized.shape != (GROUP_SIZE,):
        raise ValueError("Quantized weights must have shape (32,)")
    if scale.numel() != 1:
        raise ValueError("Scale must contain exactly one value")
    if scale.dtype != torch.float16:
        raise ValueError("Scale must use dtype float16")
    if scale.device != quantized.device:
        raise ValueError("Scale and quantized weights must use the same device")

    scale = scale.reshape(())
    if not torch.isfinite(scale) or scale < 0:
        raise ValueError("Scale must be finite and non-negative")
    return quantized.to(torch.float32) * scale.to(torch.float32)
