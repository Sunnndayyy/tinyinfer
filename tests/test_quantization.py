import pytest
import torch

from tinyinfer.quantization.int8 import dequantize_q8_group, quantize_q8_group


def test_q8_quantize_dequantize_approximately_reconstructs_weights() -> None:
    weights = torch.linspace(-2.5, 3.0, steps=32, dtype=torch.float32)

    quantized, scale = quantize_q8_group(weights)
    reconstructed = dequantize_q8_group(quantized, scale)

    # FP16 scale storage adds a small error on top of integer rounding.
    tolerance = scale.item() / 2 + weights.abs().max().item() * torch.finfo(torch.float16).eps
    torch.testing.assert_close(reconstructed, weights, rtol=0, atol=tolerance)


def test_q8_quantize_emits_the_exact_group_format() -> None:
    weights = torch.arange(-16, 16, dtype=torch.float32)

    quantized, scale = quantize_q8_group(weights)

    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert scale.shape == ()
    assert quantized.tolist() == [
        -127,
        -119,
        -111,
        -103,
        -95,
        -87,
        -79,
        -71,
        -64,
        -56,
        -48,
        -40,
        -32,
        -24,
        -16,
        -8,
        0,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        71,
        79,
        87,
        95,
        103,
        111,
        119,
    ]
    assert scale == torch.tensor(16 / 127, dtype=torch.float16)

    repeated_quantized, repeated_scale = quantize_q8_group(weights)
    assert torch.equal(repeated_quantized, quantized)
    assert torch.equal(repeated_scale, scale)


@pytest.mark.parametrize("shape", [(), (31,), (33,), (32, 2)])
def test_q8_quantize_rejects_non_one_dimensional_weights(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="one-dimensional with exactly 32"):
        quantize_q8_group(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test_q8_quantize_rejects_non_floating_weights(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="floating-point"):
        quantize_q8_group(torch.zeros(32, dtype=dtype))


@pytest.mark.parametrize("invalid_value", [torch.nan, torch.inf, -torch.inf])
def test_q8_quantize_rejects_non_finite_weights(invalid_value: float) -> None:
    weights = torch.zeros(32)
    weights[0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        quantize_q8_group(weights)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_q8_quantize_zero_group_preserves_format_and_device(device: str) -> None:
    weights = torch.zeros(32, device=device)

    quantized, scale = quantize_q8_group(weights)
    reconstructed = dequantize_q8_group(quantized, scale)

    assert quantized.device == weights.device
    assert scale.device == weights.device
    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert scale.shape == ()
    assert scale.item() == 0
    assert torch.equal(quantized, torch.zeros_like(quantized))
    assert torch.equal(reconstructed, torch.zeros_like(reconstructed))


def test_q8_dequantize_validates_quantized_tensor() -> None:
    scale = torch.tensor(0.5, dtype=torch.float16)

    with pytest.raises(ValueError, match="dtype int8"):
        dequantize_q8_group(torch.zeros(32), scale)
    with pytest.raises(ValueError, match="shape"):
        dequantize_q8_group(torch.zeros((32, 1), dtype=torch.int8), scale)


def test_q8_dequantize_rejects_scale_with_multiple_values() -> None:
    with pytest.raises(ValueError, match="exactly one value"):
        dequantize_q8_group(torch.zeros(32, dtype=torch.int8), torch.ones(2, dtype=torch.float16))


@pytest.mark.parametrize("scale", [torch.tensor(float("inf")), torch.tensor(-0.5)])
def test_q8_dequantize_rejects_invalid_scale(scale: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        dequantize_q8_group(torch.zeros(32, dtype=torch.int8), scale.to(torch.float16))


def test_q8_dequantize_rejects_wrong_scale_dtype() -> None:
    with pytest.raises(ValueError, match="dtype float16"):
        dequantize_q8_group(torch.zeros(32, dtype=torch.int8), torch.tensor(0.5))


def test_q8_dequantize_reshapes_single_value_scale_to_scalar() -> None:
    quantized = torch.arange(32, dtype=torch.int8)

    dequantized = dequantize_q8_group(quantized, torch.tensor([[0.5]], dtype=torch.float16))

    assert dequantized.shape == (32,)
    torch.testing.assert_close(dequantized, quantized.to(torch.float32) * 0.5)
