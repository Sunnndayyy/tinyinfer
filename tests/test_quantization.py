import torch

from tinyinfer.quantization.int8 import dequantize_q8_group, quantize_q8_group


def test_q8_quantize_dequantize_approximately_reconstructs_weights() -> None:
    weights = torch.linspace(-2.5, 3.0, steps=32, dtype=torch.float32)

    quantized, scale = quantize_q8_group(weights)
    reconstructed = dequantize_q8_group(quantized, scale)

    torch.testing.assert_close(reconstructed, weights, rtol=0, atol=scale.item() / 2)
