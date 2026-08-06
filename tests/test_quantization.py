import pytest
import torch

from tinyinfer.quantization.int8 import dequantize_q8_group, quantize_q8_group


def test_q8_quantize_dequantize_approximately_reconstructs_weights() -> None:
    weights = torch.linspace(-2.5, 3.0, steps=32, dtype=torch.float32)

    quantized, scale = quantize_q8_group(weights)
    reconstructed = dequantize_q8_group(quantized, scale)

    torch.testing.assert_close(reconstructed, weights, rtol=0, atol=scale.item() / 2)


@pytest.mark.parametrize("shape", [(), (32, 2)])
def test_q8_quantize_rejects_non_one_dimensional_weights(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        quantize_q8_group(torch.zeros(shape))


@pytest.mark.parametrize("invalid_value", [torch.nan, torch.inf, -torch.inf])
def test_q8_quantize_rejects_non_finite_weights(invalid_value: float) -> None:
    weights = torch.zeros(32)
    weights[0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        quantize_q8_group(weights)


def test_q8_quantize_zero_group_preserves_device() -> None:
    weights = torch.zeros(32)

    quantized, scale = quantize_q8_group(weights)

    assert quantized.device == weights.device
    assert scale.device == weights.device


def test_q8_dequantize_validates_quantized_tensor() -> None:
    scale = torch.tensor(0.5)

    with pytest.raises(ValueError, match="dtype int8"):
        dequantize_q8_group(torch.zeros(32), scale)
    with pytest.raises(ValueError, match="shape"):
        dequantize_q8_group(torch.zeros((32, 1), dtype=torch.int8), scale)


def test_q8_dequantize_rejects_scale_with_multiple_values() -> None:
    with pytest.raises(ValueError, match="exactly one value"):
        dequantize_q8_group(torch.zeros(32, dtype=torch.int8), torch.ones(2))


def test_q8_dequantize_reshapes_single_value_scale_to_scalar() -> None:
    quantized = torch.arange(32, dtype=torch.int8)

    dequantized = dequantize_q8_group(quantized, torch.tensor([[0.5]]))

    assert dequantized.shape == (32,)
    torch.testing.assert_close(dequantized, quantized.to(torch.float32) * 0.5)
