"""4 bit quantization implementation"""

from torch import Tensor
import torch

GROUP_SIZE = 32
Q4_MAX = 15


def quantize_q4_group(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize 32 floating-point weights into INT4 values and a scale factor"""

    # validate input tensor is 32 weights
    if tensor.shape[0] != GROUP_SIZE:
        raise ValueError(f"Input tensor must have 32 weights, but got {tensor.shape[0]}")

    working = tensor.to(torch.float32)
    largest_abs = working.abs().max()

    # handle all zero weights, return all zeros and a scale of 1.0
    if largest_abs == 0:
        return torch.zeros(GROUP_SIZE, dtype=torch.int4), torch.tensor(1.0, dtype=torch.float32)

    # calculate the scale factor to normalise the tensor to the range [-Q4_MAX, Q4_MAX]
    scale = largest_abs / Q4_MAX
    quantized = (working / scale).round().clamp(-Q4_MAX, Q4_MAX).to(torch.int4)

    return quantized, scale.to(torch.float32)

def dequantize_q4_group(quantized: Tensor, scale: Tensor) -> Tensor:
    """Approximately restore a group of Q4 weights."""
    return quantized.to(torch.float32) * scale.to(torch.float32)

