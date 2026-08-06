import pytest
import torch

from tinyinfer.quantization import (
    dequantize_q8,
    dequantize_q8_group,
    q8_linear_reference,
    quantize_q8,
    quantize_q8_group,
)


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

    with pytest.raises(ValueError, match="shape"):
        dequantize_q8_group(torch.zeros((32, 1), dtype=torch.int8), scale)


def test_q8_dequantize_requires_scalar_scale() -> None:
    with pytest.raises(ValueError, match="scalar"):
        dequantize_q8_group(
            torch.zeros(32, dtype=torch.int8),
            torch.ones((1,), dtype=torch.float16),
        )


def test_q8_matrix_groups_each_row_from_left_to_right() -> None:
    groups = [
        torch.arange(-16, 16, dtype=torch.float32),
        torch.zeros(32),
        torch.linspace(-1, 1, steps=32),
        torch.linspace(-100, 50, steps=32),
    ]
    weights = torch.stack((torch.cat(groups[:2]), torch.cat(groups[2:])))

    quantized, scales = quantize_q8(weights)

    assert quantized.shape == weights.shape
    assert quantized.dtype == torch.int8
    assert scales.shape == (2, 2)
    assert scales.dtype == torch.float16

    # Exact group comparisons prove the matrix layout, not just its approximation error.
    for row in range(2):
        for group in range(2):
            expected_values, expected_scale = quantize_q8_group(groups[row * 2 + group])
            start = group * 32
            assert torch.equal(quantized[row, start : start + 32], expected_values)
            assert torch.equal(scales[row, group], expected_scale)


def test_q8_matrix_dequantize_restores_the_group_layout() -> None:
    weights = torch.linspace(-5, 3, steps=128).reshape(2, 64)
    quantized, scales = quantize_q8(weights)

    restored = dequantize_q8(quantized, scales)

    expected_groups = []
    for row in range(2):
        row_groups = []
        for group in range(2):
            start = group * 32
            row_groups.append(
                dequantize_q8_group(quantized[row, start : start + 32], scales[row, group])
            )
        expected_groups.append(torch.cat(row_groups))
    assert torch.equal(restored, torch.stack(expected_groups))


@pytest.mark.parametrize("shape", [(64,), (2, 2, 32), (2, 63), (2, 0)])
def test_q8_matrix_quantize_rejects_unsupported_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        quantize_q8(torch.zeros(shape))


def test_q8_matrix_quantize_rejects_non_floating_weights() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        quantize_q8(torch.zeros((2, 64), dtype=torch.int32))


@pytest.mark.parametrize("invalid_value", [torch.nan, torch.inf, -torch.inf])
def test_q8_matrix_quantize_rejects_non_finite_weights(invalid_value: float) -> None:
    weights = torch.zeros((2, 64))
    weights[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        quantize_q8(weights)


@pytest.mark.parametrize("value", [1e-6, 9e6])
def test_q8_quantize_rejects_scales_that_float16_cannot_represent(value: float) -> None:
    weights = torch.full((2, 64), value)

    with pytest.raises(ValueError, match="representable"):
        quantize_q8(weights)


def test_q8_quantize_rejects_values_that_overflow_float32() -> None:
    weights = torch.full((2, 64), 1e300, dtype=torch.float64)

    with pytest.raises(ValueError, match="finite"):
        quantize_q8(weights)


@pytest.mark.parametrize(
    ("value", "dtype", "message"),
    [
        (1e-6, torch.float32, "representable"),
        (9e6, torch.float32, "representable"),
        (1e300, torch.float64, "finite"),
    ],
)
def test_q8_group_rejects_unrepresentable_values(
    value: float,
    dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quantize_q8_group(torch.full((32,), value, dtype=dtype))


def test_q8_matrix_dequantize_rejects_malformed_scales() -> None:
    quantized = torch.zeros((2, 64), dtype=torch.int8)

    with pytest.raises(ValueError, match="shape"):
        dequantize_q8(quantized, torch.zeros((2, 1), dtype=torch.float16))


@pytest.mark.parametrize(
    "quantized",
    [
        torch.zeros(64, dtype=torch.int8),
        torch.zeros((2, 0), dtype=torch.int8),
        torch.zeros((2, 63), dtype=torch.int8),
    ],
)
def test_q8_matrix_dequantize_rejects_malformed_weights(quantized: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        dequantize_q8(quantized, torch.zeros((2, 2), dtype=torch.float16))


def test_q8_matrix_quantization_is_deterministic() -> None:
    weights = torch.linspace(-7, 11, steps=128).reshape(2, 64)

    first = quantize_q8(weights)
    second = quantize_q8(weights)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_q8_matrix_preserves_device(device: str) -> None:
    weights = torch.linspace(-2, 3, steps=128, device=device).reshape(2, 64)

    quantized, scales = quantize_q8(weights)
    restored = dequantize_q8(quantized, scales)

    assert quantized.device == weights.device
    assert scales.device == weights.device
    assert restored.device == weights.device


@pytest.mark.parametrize("shape", [(64,), (3, 64), (2, 3, 64)])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_q8_linear_reference_matches_explicit_dequantize_and_linear(
    shape: tuple[int, ...],
    use_bias: bool,
    input_dtype: torch.dtype,
) -> None:
    weights = torch.linspace(-4, 6, steps=256).reshape(4, 64)
    inputs = torch.linspace(-2, 3, steps=torch.tensor(shape).prod().item()).reshape(shape)
    inputs = inputs.to(input_dtype)
    bias = torch.linspace(-0.5, 0.5, steps=4).to(input_dtype) if use_bias else None
    quantized, scales = quantize_q8(weights)

    actual = q8_linear_reference(inputs, quantized, scales, bias)
    grouped = quantized.reshape(4, 2, 32).to(torch.float32)
    restored = (grouped * scales.to(torch.float32).unsqueeze(-1)).reshape(4, 64)
    expected = torch.matmul(inputs.to(torch.float32), restored.T)
    if bias is not None:
        expected += bias.to(torch.float32)

    assert actual.shape == (*shape[:-1], 4)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_q8_linear_reference_rejects_wrong_input_width() -> None:
    quantized, scales = quantize_q8(torch.zeros((4, 64)))

    with pytest.raises(ValueError, match="last dimension"):
        q8_linear_reference(torch.zeros((2, 63)), quantized, scales)
