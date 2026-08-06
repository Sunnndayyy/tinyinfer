"""CPU reference and fused MPS linear operations for packed weights."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .int8 import dequantize_q8
from .metal import _q8_linear_mps, q8_linear_mps


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


def q8_linear(
    inputs: Tensor,
    quantized_weights: Tensor,
    scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Use the readable CPU oracle or fused MPS operation for the same packed format."""
    if inputs.device.type == "mps":
        return q8_linear_mps(inputs, quantized_weights, scales, bias)
    return q8_linear_reference(inputs, quantized_weights, scales, bias)


def _q8_linear_module(
    inputs: Tensor,
    quantized_weights: Tensor,
    scales: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    if inputs.device.type == "mps":
        if (
            inputs.dtype != torch.bfloat16
            or scales.dtype != torch.float16
            or (bias is not None and bias.dtype != torch.bfloat16)
        ):
            raise ValueError("Q8 MPS modules require bfloat16 activations/bias and float16 scales")
        return _q8_linear_mps(inputs, quantized_weights, scales, bias)
    return q8_linear_reference(inputs, quantized_weights, scales, bias)
