"""Slow reference linear operations for packed weights."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .int8 import dequantize_q8


def q8_linear_reference(
    inputs: Tensor,
    quantized_weights: Tensor,
    scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Run an FP32 linear operation after explicitly restoring Q8 weights."""
    weights = dequantize_q8(quantized_weights, scales)
    if inputs.ndim == 0 or inputs.shape[-1] != weights.shape[1]:
        raise ValueError("Input last dimension must match the quantized weight width")
    if not inputs.is_floating_point():
        raise ValueError("Inputs must use a floating-point dtype")

    # This full dequantization is intentionally obvious: it is an oracle, never a hot path.
    reference_bias = None if bias is None else bias.to(torch.float32)
    return F.linear(inputs.to(torch.float32), weights, reference_bias)
