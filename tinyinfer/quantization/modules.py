"""Inference modules backed by readable Q8 reference operations."""

import torch
from torch import Tensor, nn

from .int8 import GROUP_SIZE, dequantize_q8, quantize_q8
from .linear import q8_linear_reference


def _empty_q8_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    rows, columns = weight.shape
    packed = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty((rows, columns // GROUP_SIZE), device=weight.device, dtype=torch.float16)
    return packed, scales


class Q8Linear(nn.Module):
    def __init__(self, weight: Tensor, scales: Tensor, bias: Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "Q8Linear":
        weight, scales = quantize_q8(linear.weight.detach())
        bias = None if linear.bias is None else linear.bias.detach().clone()
        return cls(weight, scales, bias)

    @classmethod
    def empty_like(cls, linear: nn.Linear) -> "Q8Linear":
        weight, scales = _empty_q8_weight(linear.weight)
        bias = None if linear.bias is None else torch.empty_like(linear.bias)
        return cls(weight, scales, bias)

    def forward(self, inputs: Tensor) -> Tensor:
        output = q8_linear_reference(inputs, self.weight, self.scales, self.bias)
        return output.to(inputs.dtype)


class Q8Embedding(nn.Module):
    def __init__(self, weight: Tensor, scales: Tensor, output_dtype: torch.dtype):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        # A buffer follows module dtype conversions without adding checkpoint data.
        self.register_buffer(
            "_output_dtype",
            torch.empty(0, dtype=output_dtype, device="cpu"),
            persistent=False,
        )

    @classmethod
    def from_float(cls, embedding: nn.Embedding) -> "Q8Embedding":
        weight, scales = quantize_q8(embedding.weight.detach())
        return cls(weight, scales, embedding.weight.dtype)

    @classmethod
    def empty_like(cls, embedding: nn.Embedding, output_dtype: torch.dtype) -> "Q8Embedding":
        weight, scales = _empty_q8_weight(embedding.weight)
        return cls(weight, scales, output_dtype)

    def forward(self, input_ids: Tensor) -> Tensor:
        width = self.weight.shape[1]
        weights = self.weight[input_ids].reshape(-1, width)
        scales = self.scales[input_ids].reshape(-1, self.scales.shape[1])
        shape = (*input_ids.shape, width)
        return dequantize_q8(weights, scales).reshape(shape).to(self._output_dtype.dtype)

    def project(self, inputs: Tensor) -> Tensor:
        """Reuse the tied embedding weights for the vocabulary projection."""
        output = q8_linear_reference(inputs, self.weight, self.scales)
        return output.to(inputs.dtype)
