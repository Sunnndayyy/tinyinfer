from importlib import resources

import pytest
import torch

from tinyinfer.quantization import Q8Embedding, Q8Linear, metal, q8_linear, quantize_q8
from tinyinfer.quantization import linear as linear_module
from tinyinfer.quantization import modules as modules_module

MPS_AVAILABLE = metal.q8_mps_available()


def test_q8_mps_capability_requires_mps_and_compile_shader(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert not metal.q8_mps_available()

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.delattr(torch.mps, "compile_shader")
    assert not metal.q8_mps_available()

    with pytest.raises(RuntimeError, match="torch.mps.compile_shader"):
        metal.require_q8_mps()


def test_q8_metal_source_is_packaged() -> None:
    source = resources.files("tinyinfer.quantization").joinpath("weight_only.metal").read_text()

    assert "kernel void q8_linear_bf16" in source


def test_q8_shader_compiles_once(monkeypatch) -> None:
    calls = []
    metal._shader_library.cache_clear()
    monkeypatch.setattr(
        torch.mps, "compile_shader", lambda source: calls.append(source) or object()
    )

    assert metal._shader_library() is metal._shader_library()
    assert len(calls) == 1
    metal._shader_library.cache_clear()


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
def test_q8_metal_linear_matches_a_hand_calculated_matrix() -> None:
    inputs = torch.arange(1, 33, dtype=torch.bfloat16, device="mps")
    weights = torch.stack((torch.ones(32), -torch.ones(32))).to(torch.int8).to("mps")
    scales = torch.tensor([[0.5], [0.25]], dtype=torch.float16, device="mps")

    actual = q8_linear(inputs, weights, scales)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.cpu(),
        torch.tensor([264.0, -132.0], dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
@pytest.mark.parametrize(
    ("input_shape", "output_width"),
    [
        ((4, 64), 7),
        ((2, 2, 64), 9),
        ((1, 1536), 256),
        ((1, 1536), 1536),
        ((1, 1536), 8960),
        ((1, 8960), 1536),
        ((1, 1536), 4096),
    ],
)
def test_q8_metal_linear_matches_the_packed_weight_oracle(
    input_shape: tuple[int, ...], output_width: int
) -> None:
    generator = torch.Generator().manual_seed(7)
    input_width = input_shape[-1]
    inputs = torch.randn(input_shape, generator=generator, dtype=torch.bfloat16)
    weights = torch.randint(-127, 128, (output_width, input_width), generator=generator).to(
        torch.int8
    )
    scales = (
        torch.rand((output_width, input_width // 32), generator=generator, dtype=torch.float16)
        / 127
    )
    expected = q8_linear(inputs, weights, scales).to(torch.bfloat16)

    actual = q8_linear(inputs.to("mps"), weights.to("mps"), scales.to("mps")).cpu()

    assert actual.shape == (*input_shape[:-1], output_width)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
def test_q8_metal_linear_fuses_bias() -> None:
    inputs = torch.ones((2, 32), dtype=torch.bfloat16, device="mps")
    weights = torch.ones((4, 32), dtype=torch.int8, device="mps")
    scales = torch.ones((4, 1), dtype=torch.float16, device="mps")
    bias = torch.arange(4, dtype=torch.bfloat16, device="mps")

    actual = q8_linear(inputs, weights, scales, bias)

    expected = torch.tensor([[32, 33, 34, 35]] * 2, dtype=torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
def test_q8_metal_linear_never_calls_the_full_dequantizer(monkeypatch) -> None:
    monkeypatch.setattr(
        linear_module,
        "dequantize_q8",
        lambda *args: pytest.fail("the MPS path must not restore the weight matrix"),
    )
    inputs = torch.ones((1, 32), dtype=torch.bfloat16, device="mps")
    weights = torch.ones((2, 32), dtype=torch.int8, device="mps")
    scales = torch.ones((2, 1), dtype=torch.float16, device="mps")

    q8_linear(inputs, weights, scales)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
def test_q8_mps_embedding_never_calls_the_full_dequantizer(monkeypatch) -> None:
    source = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32) / 16
    weights, scales = quantize_q8(source)
    embedding = Q8Embedding(weights.to("mps"), scales.to("mps"), torch.bfloat16)
    input_ids = torch.tensor([[0, 3], [7, 3]], device="mps")
    selected_weights = weights[input_ids.cpu()].reshape(-1, 32)
    selected_scales = scales[input_ids.cpu()].reshape(-1, 1)
    expected = modules_module.dequantize_q8(selected_weights, selected_scales)
    monkeypatch.setattr(
        modules_module,
        "dequantize_q8",
        lambda *args: pytest.fail("MPS embedding must read packed rows directly"),
    )

    actual = embedding(input_ids).cpu()

    torch.testing.assert_close(actual, expected.reshape(2, 2, 32).to(torch.bfloat16))


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
@pytest.mark.parametrize(
    ("input_dtype", "module_dtype", "scale_dtype"),
    [
        (torch.float16, torch.bfloat16, torch.float16),
        (torch.float32, torch.bfloat16, torch.float16),
        (torch.bfloat16, torch.float16, torch.float16),
        (torch.bfloat16, torch.float32, torch.float16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ],
)
def test_q8_mps_linear_module_rejects_unsupported_dtypes(
    input_dtype: torch.dtype, module_dtype: torch.dtype, scale_dtype: torch.dtype
) -> None:
    module = Q8Linear.from_float(torch.nn.Linear(32, 2, dtype=module_dtype)).to("mps")
    module.scales = module.scales.to(scale_dtype)
    inputs = torch.zeros((1, 32), dtype=input_dtype, device="mps")

    with pytest.raises(ValueError, match="bfloat16 activations/bias and float16 scales"):
        module(inputs)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
def test_q8_mps_embedding_rejects_non_bfloat16_output_dtype() -> None:
    embedding = Q8Embedding.from_float(torch.nn.Embedding(8, 32, dtype=torch.float16)).to("mps")
    input_ids = torch.tensor([0], device="mps")

    with pytest.raises(ValueError, match="bfloat16 output dtype"):
        embedding(input_ids)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
def test_q8_metal_linear_normalizes_non_contiguous_inputs() -> None:
    inputs = torch.arange(128, dtype=torch.bfloat16, device="mps").reshape(64, 2).T
    weights = torch.ones((4, 64), dtype=torch.int8, device="mps")
    scales = torch.ones((4, 2), dtype=torch.float16, device="mps")

    actual = q8_linear(inputs, weights, scales)
    expected = q8_linear(inputs.cpu(), weights.cpu(), scales.cpu()).to(torch.bfloat16)

    assert not inputs.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires Apple MPS custom shaders")
@pytest.mark.parametrize(
    ("inputs", "weights", "scales", "message"),
    [
        (
            lambda: torch.zeros((1, 32), dtype=torch.float16, device="mps"),
            lambda: torch.zeros((2, 32), dtype=torch.int8, device="mps"),
            lambda: torch.zeros((2, 1), dtype=torch.float16, device="mps"),
            "bfloat16",
        ),
        (
            lambda: torch.zeros((1, 32), dtype=torch.bfloat16, device="mps"),
            lambda: torch.zeros((2, 32), dtype=torch.int16, device="mps"),
            lambda: torch.zeros((2, 1), dtype=torch.float16, device="mps"),
            "int8",
        ),
        (
            lambda: torch.zeros((1, 32), dtype=torch.bfloat16, device="mps"),
            lambda: torch.zeros((2, 32), dtype=torch.int8, device="mps"),
            lambda: torch.zeros((2, 1), dtype=torch.float32, device="mps"),
            "float16",
        ),
        (
            lambda: torch.zeros((1, 64), dtype=torch.bfloat16, device="mps"),
            lambda: torch.zeros((2, 32), dtype=torch.int8, device="mps"),
            lambda: torch.zeros((2, 1), dtype=torch.float16, device="mps"),
            "width",
        ),
        (
            lambda: torch.zeros((1, 32), dtype=torch.bfloat16, device="mps"),
            lambda: torch.zeros((2, 32), dtype=torch.int8, device="mps"),
            lambda: torch.zeros((2, 2), dtype=torch.float16, device="mps"),
            "shape",
        ),
    ],
)
def test_q8_metal_linear_rejects_invalid_inputs(inputs, weights, scales, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        q8_linear(inputs(), weights(), scales())
