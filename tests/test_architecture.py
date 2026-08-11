import json
from dataclasses import fields, is_dataclass, replace
from enum import Enum

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tinyinfer.architecture import (
    Architecture,
    DenseSwiGLUSpec,
    GatedDeltaNetSpec,
    MultiaxisRoPESpec,
    NormMode,
    NormSpec,
    ScalarRoPESpec,
    SoftmaxAttentionSpec,
    load_model_spec,
)
from tinyinfer.model import Attention, QwenConfig, QwenForCausalLM, RMSNorm


def qwen2_values() -> dict:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }


def qwen3_values() -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 48,
        "hidden_size": 24,
        "intermediate_size": 40,
        "num_hidden_layers": 3,
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "rope_scaling": None,
        "hidden_act": "silu",
        "attention_bias": False,
    }


def qwen3_5_values() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 3,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "attn_output_gate": True,
            "hidden_act": "silu",
            "attention_bias": False,
            "mamba_ssm_dtype": "float32",
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_value_head_dim": 8,
            "use_sliding_window": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
    }


def write_config(tmp_path, values: dict, *, generation_eos=None) -> None:
    tmp_path.joinpath("config.json").write_text(json.dumps(values))
    if generation_eos is not None:
        tmp_path.joinpath("generation_config.json").write_text(
            json.dumps({"eos_token_id": generation_eos})
        )


def test_qwen2_config_becomes_an_ordered_dense_descriptor(tmp_path) -> None:
    write_config(tmp_path, qwen2_values(), generation_eos=[2, 7, 8])

    spec = load_model_spec(tmp_path)

    assert spec.architecture is Architecture.QWEN2
    assert spec.num_hidden_layers == 2
    assert spec.stop_token_ids == frozenset({2, 7, 8})
    assert isinstance(spec.position, ScalarRoPESpec)
    assert spec.position.rotary_fraction == 1.0
    assert spec.final_norm.mode is NormMode.DIRECT
    assert spec.output.tie_embeddings is True
    assert all(isinstance(layer.token_mixer, SoftmaxAttentionSpec) for layer in spec.layers)
    assert all(layer.token_mixer.qkv_bias for layer in spec.layers)
    assert all(layer.token_mixer.qk_norm is None for layer in spec.layers)
    assert all(isinstance(layer.channel_mixer, DenseSwiGLUSpec) for layer in spec.layers)


def test_qwen3_config_selects_qk_norm_and_bias_free_attention(tmp_path) -> None:
    write_config(tmp_path, qwen3_values())

    spec = load_model_spec(tmp_path)
    attention = spec.layers[0].token_mixer

    assert spec.architecture is Architecture.QWEN3
    assert isinstance(attention, SoftmaxAttentionSpec)
    assert attention.qkv_bias is False
    assert attention.output_gate is False
    assert attention.qk_norm is not None
    assert attention.qk_norm.mode is NormMode.DIRECT
    assert attention.head_dim == 8
    assert spec.output.tie_embeddings is False
    assert isinstance(spec.position, ScalarRoPESpec)


def test_qwen3_5_nested_text_config_expands_the_exact_layer_order(tmp_path) -> None:
    write_config(tmp_path, qwen3_5_values())

    spec = load_model_spec(tmp_path)

    assert spec.architecture is Architecture.QWEN3_5
    assert spec.final_norm.mode is NormMode.UNIT_OFFSET
    assert [type(layer.token_mixer) for layer in spec.layers] == [
        GatedDeltaNetSpec,
        GatedDeltaNetSpec,
        GatedDeltaNetSpec,
        SoftmaxAttentionSpec,
    ]
    delta = spec.layers[0].token_mixer
    assert isinstance(delta, GatedDeltaNetSpec)
    assert (delta.key_heads, delta.value_heads) == (4, 8)
    assert delta.conv_kernel == 4
    attention = spec.layers[-1].token_mixer
    assert isinstance(attention, SoftmaxAttentionSpec)
    assert attention.output_gate is True
    assert attention.qk_norm is not None
    assert attention.qk_norm.mode is NormMode.UNIT_OFFSET
    assert isinstance(spec.position, MultiaxisRoPESpec)
    assert spec.position.rotary_fraction == 0.5
    assert spec.position.sections == (2, 1, 1)
    assert spec.position.interleaved is True


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ("qwen3_next", "Qwen3NextForCausalLM"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("llama", "LlamaForCausalLM"),
    ],
)
def test_descriptor_rejects_architectures_outside_the_dense_scope(
    tmp_path, model_type: str, architecture: str
) -> None:
    write_config(tmp_path, {"model_type": model_type, "architectures": [architecture]})

    with pytest.raises(ValueError, match="supported dense architectures"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_inconsistent_qwen3_5_layer_count(tmp_path) -> None:
    values = qwen3_5_values()
    values["text_config"]["layer_types"] = ["full_attention"]
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="layer_types"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_a_missing_required_dense_field(tmp_path) -> None:
    values = qwen3_values()
    values.pop("hidden_size")
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="hidden_size"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_incompatible_attention_head_groups(tmp_path) -> None:
    values = qwen3_values()
    values["num_key_value_heads"] = 4
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="divisible"):
        load_model_spec(tmp_path)


def test_qwen3_5_dense_descriptor_rejects_expert_fields(tmp_path) -> None:
    values = qwen3_5_values()
    values["text_config"]["num_experts"] = 8
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="MoE fields"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_invalid_qwen3_5_rotary_sections(tmp_path) -> None:
    values = qwen3_5_values()
    values["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 1]
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="mrope_section"):
        load_model_spec(tmp_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("rope_scaling", {"rope_type": "yarn"}), ("use_sliding_window", True)],
)
def test_descriptor_rejects_unsupported_qwen3_execution_options(tmp_path, key, value) -> None:
    values = qwen3_values()
    values[key] = value
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match=key):
        load_model_spec(tmp_path)


@pytest.mark.parametrize(
    ("family", "field", "value", "error"),
    [
        ("qwen2", "hidden_act", "gelu", "hidden_act"),
        ("qwen2", "attention_bias", False, "attention_bias"),
        ("qwen3", "hidden_act", "gelu", "hidden_act"),
        ("qwen3", "attention_bias", True, "attention_bias"),
        ("qwen3_5", "hidden_act", "gelu", "hidden_act"),
        ("qwen3_5", "attention_bias", True, "attention_bias"),
        ("qwen3_5", "mamba_ssm_dtype", "float16", "mamba_ssm_dtype"),
    ],
)
def test_descriptor_rejects_config_that_disagrees_with_selected_computation(
    tmp_path, family, field, value, error
) -> None:
    values = {
        "qwen2": qwen2_values,
        "qwen3": qwen3_values,
        "qwen3_5": qwen3_5_values,
    }[family]()
    target = values["text_config"] if family == "qwen3_5" else values
    target[field] = value
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match=error):
        load_model_spec(tmp_path)


@pytest.mark.parametrize("generation_eos", [-1, [2, -1]])
def test_descriptor_rejects_negative_generation_stop_ids(tmp_path, generation_eos) -> None:
    write_config(tmp_path, qwen2_values(), generation_eos=generation_eos)

    with pytest.raises(ValueError, match="non-negative"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_non_finite_positive_numbers(tmp_path) -> None:
    values = qwen3_values()
    values["rms_norm_eps"] = float("nan")
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="rms_norm_eps"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_non_string_layer_types(tmp_path) -> None:
    values = qwen3_5_values()
    values["text_config"]["layer_types"][0] = ["linear_attention"]
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="layer_types"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_non_string_architecture_names(tmp_path) -> None:
    values = qwen2_values()
    values["architectures"] = [["Qwen2ForCausalLM"]]
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="architecture name"):
        load_model_spec(tmp_path)


def test_descriptor_rejects_boolean_rotary_sections(tmp_path) -> None:
    values = qwen3_5_values()
    values["text_config"]["rope_parameters"]["mrope_section"] = [True, 1, 2]
    write_config(tmp_path, values)

    with pytest.raises(ValueError, match="mrope_section"):
        load_model_spec(tmp_path)


def test_descriptor_contains_only_frozen_data(tmp_path) -> None:
    write_config(tmp_path, qwen3_5_values())
    spec = load_model_spec(tmp_path)

    def assert_data(value) -> None:
        assert not isinstance(value, nn.Module)
        assert not callable(value)
        if is_dataclass(value):
            for field in fields(value):
                assert_data(getattr(value, field.name))
        elif isinstance(value, (tuple, frozenset)):
            for item in value:
                assert_data(item)
        else:
            assert isinstance(value, (Enum, str, int, float, bool, type(None)))

    assert_data(spec)


def test_qwen2_descriptor_preserves_existing_model_construction(tmp_path) -> None:
    values = qwen2_values()
    write_config(tmp_path, values)
    spec = load_model_spec(tmp_path)
    direct_config = QwenConfig(
        **{
            key: value
            for key, value in values.items()
            if key not in {"architectures", "model_type", "hidden_act"}
        }
    )
    descriptor_config = QwenConfig.from_model_spec(spec)

    torch.manual_seed(41)
    direct = QwenForCausalLM(direct_config).eval()
    torch.manual_seed(41)
    descriptor_backed = QwenForCausalLM(descriptor_config).eval()
    input_ids = torch.tensor([[1, 5, 7]])

    assert direct.state_dict().keys() == descriptor_backed.state_dict().keys()
    with torch.inference_mode():
        torch.testing.assert_close(direct(input_ids), descriptor_backed(input_ids))


def test_tied_qwen3_descriptor_builds_its_dense_runtime(tmp_path) -> None:
    values = qwen3_values()
    values["tie_word_embeddings"] = True
    write_config(tmp_path, values)

    config = QwenConfig.from_directory(tmp_path)
    model = QwenForCausalLM(config).eval()
    attention = model.model.layers[0].self_attn

    assert config.architecture is Architecture.QWEN3
    assert config.head_dim == 8
    assert isinstance(attention, Attention)
    assert attention.q_proj.bias is None
    assert attention.k_proj.bias is None
    assert attention.v_proj.bias is None
    assert isinstance(attention.q_norm, RMSNorm)
    assert isinstance(attention.k_norm, RMSNorm)
    assert attention.q_norm.weight.shape == (8,)
    assert attention.k_norm.weight.shape == (8,)
    assert model(torch.tensor([[1, 5, 7]])).shape == (1, 3, 48)


def test_tied_qwen3_checkpoint_loader_restores_qk_norm_weights(tmp_path) -> None:
    values = qwen3_values()
    values["tie_word_embeddings"] = True
    write_config(tmp_path, values)
    source = QwenForCausalLM(QwenConfig.from_directory(tmp_path)).eval()
    tensors = {name: value.detach().clone() for name, value in source.state_dict().items()}
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, tmp_path / "model.safetensors")

    loaded = QwenForCausalLM.from_pretrained(
        tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert loaded.training is False
    loaded_state = loaded.state_dict()
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(loaded_state[name], expected)


def test_qwen3_construction_rejects_unit_offset_qk_norm(tmp_path) -> None:
    values = qwen3_values()
    values["tie_word_embeddings"] = True
    write_config(tmp_path, values)
    spec = load_model_spec(tmp_path)
    attention = replace(
        spec.layers[0].token_mixer,
        qk_norm=NormSpec(NormMode.UNIT_OFFSET, 1e-6),
    )
    layer = replace(spec.layers[0], token_mixer=attention)

    with pytest.raises(NotImplementedError, match="unit-offset attention Q/K RMSNorm"):
        QwenConfig.from_model_spec(replace(spec, layers=(layer, *spec.layers[1:])))


def test_qwen3_construction_rejects_untied_output(tmp_path) -> None:
    write_config(tmp_path, qwen3_values())

    with pytest.raises(NotImplementedError, match="untied output projection"):
        QwenConfig.from_directory(tmp_path)


def test_qwen3_5_construction_fails_at_its_first_unimplemented_capability(tmp_path) -> None:
    write_config(tmp_path, qwen3_5_values())
    spec = load_model_spec(tmp_path)

    with pytest.raises(NotImplementedError, match="unit-offset RMSNorm"):
        QwenConfig.from_model_spec(spec)


def test_qwen2_construction_rejects_inconsistent_layer_norms(tmp_path) -> None:
    write_config(tmp_path, qwen2_values())
    spec = load_model_spec(tmp_path)
    layer = replace(spec.layers[0], input_norm=NormSpec(NormMode.DIRECT, 1e-5))
    inconsistent = replace(spec, layers=(layer, *spec.layers[1:]))

    with pytest.raises(ValueError, match="normalization"):
        QwenConfig.from_model_spec(inconsistent)
