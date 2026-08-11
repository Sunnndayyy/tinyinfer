from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias


class Architecture(str, Enum):
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    QWEN3_5 = "qwen3_5"


class NormMode(str, Enum):
    DIRECT = "direct"
    UNIT_OFFSET = "unit_offset"


@dataclass(frozen=True)
class NormSpec:
    mode: NormMode
    eps: float


@dataclass(frozen=True)
class ScalarRoPESpec:
    theta: float
    rotary_fraction: float = 1.0


@dataclass(frozen=True)
class MultiaxisRoPESpec:
    theta: float
    rotary_fraction: float
    sections: tuple[int, int, int]
    interleaved: bool


PositionSpec: TypeAlias = ScalarRoPESpec | MultiaxisRoPESpec


@dataclass(frozen=True)
class SoftmaxAttentionSpec:
    query_heads: int
    key_value_heads: int
    head_dim: int
    qkv_bias: bool
    qk_norm: NormSpec | None
    output_gate: bool

    @property
    def key_value_groups(self) -> int:
        return self.query_heads // self.key_value_heads


@dataclass(frozen=True)
class GatedDeltaNetSpec:
    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int


TokenMixerSpec: TypeAlias = SoftmaxAttentionSpec | GatedDeltaNetSpec


@dataclass(frozen=True)
class DenseSwiGLUSpec:
    intermediate_size: int


@dataclass(frozen=True)
class LayerSpec:
    input_norm: NormSpec
    post_attention_norm: NormSpec
    token_mixer: TokenMixerSpec
    channel_mixer: DenseSwiGLUSpec


@dataclass(frozen=True)
class OutputSpec:
    tie_embeddings: bool


@dataclass(frozen=True)
class ModelSpec:
    architecture: Architecture
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    layers: tuple[LayerSpec, ...]
    final_norm: NormSpec
    position: PositionSpec
    output: OutputSpec
    bos_token_id: int | None
    eos_token_id: int
    pad_token_id: int | None = None
    additional_eos_token_ids: tuple[int, ...] = ()

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers)

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return frozenset((self.eos_token_id, *self.additional_eos_token_ids))


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read a valid JSON object from {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _required(values: dict, name: str, context: str):
    if name not in values:
        raise ValueError(f"{context} is missing required field {name!r}")
    return values[name]


def _positive_int(values: dict, name: str, context: str) -> int:
    value = _required(values, name, context)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context}.{name} must be a positive integer")
    return value


def _positive_float(values: dict, name: str, context: str) -> float:
    value = _required(values, name, context)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{context}.{name} must be positive")
    return float(value)


def _required_bool(values: dict, name: str, context: str) -> bool:
    value = _required(values, name, context)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{name} must be a boolean")
    return value


def _require_exact(values: dict, name: str, expected, context: str, *, optional=False) -> None:
    if optional and name not in values:
        return
    value = _required(values, name, context)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{context}.{name} must be {expected!r}")


def _token_id(values: dict, name: str, context: str) -> int:
    value = _required(values, name, context)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context}.{name} must be a non-negative integer")
    return value


def _optional_token_id(values: dict, name: str, context: str) -> int | None:
    value = values.get(name)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{context}.{name} must be a non-negative integer or null")
    return value


def _validate_attention_dimensions(query_heads: int, key_value_heads: int, context: str) -> None:
    if query_heads % key_value_heads:
        raise ValueError(f"{context}.num_attention_heads must be divisible by num_key_value_heads")


def _validate_execution_options(values: dict, context: str) -> None:
    if values.get("rope_scaling") is not None:
        raise ValueError(f"{context}.rope_scaling is not supported")
    if "use_sliding_window" in values and not isinstance(values["use_sliding_window"], bool):
        raise ValueError(f"{context}.use_sliding_window must be a boolean")
    if values.get("use_sliding_window", False):
        raise ValueError(f"{context}.use_sliding_window is not supported")


def _generation_stop_ids(model_dir: Path, eos_token_id: int) -> tuple[int, ...]:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return ()
    value = _read_object(path).get("eos_token_id", [])
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        candidates = [value]
    elif isinstance(value, list) and all(
        isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        for token_id in value
    ):
        candidates = value
    else:
        raise ValueError(
            "generation_config.json eos_token_id must be a non-negative integer or list"
        )
    return tuple(dict.fromkeys(token_id for token_id in candidates if token_id != eos_token_id))


def _dense_layer(norm: NormSpec, token_mixer: TokenMixerSpec, intermediate_size: int) -> LayerSpec:
    return LayerSpec(
        input_norm=norm,
        post_attention_norm=norm,
        token_mixer=token_mixer,
        channel_mixer=DenseSwiGLUSpec(intermediate_size),
    )


def _parse_qwen2(values: dict, model_dir: Path) -> ModelSpec:
    context = "config"
    _validate_execution_options(values, context)
    _require_exact(values, "hidden_act", "silu", context, optional=True)
    _require_exact(values, "attention_bias", True, context, optional=True)
    hidden_size = _positive_int(values, "hidden_size", context)
    query_heads = _positive_int(values, "num_attention_heads", context)
    key_value_heads = _positive_int(values, "num_key_value_heads", context)
    _validate_attention_dimensions(query_heads, key_value_heads, context)
    if hidden_size % query_heads:
        raise ValueError("config.hidden_size must be divisible by num_attention_heads")
    head_dim = hidden_size // query_heads
    norm = NormSpec(NormMode.DIRECT, _positive_float(values, "rms_norm_eps", context))
    attention = SoftmaxAttentionSpec(
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        qkv_bias=True,
        qk_norm=None,
        output_gate=False,
    )
    layer = _dense_layer(
        norm,
        attention,
        _positive_int(values, "intermediate_size", context),
    )
    eos_token_id = _token_id(values, "eos_token_id", context)
    return ModelSpec(
        architecture=Architecture.QWEN2,
        vocab_size=_positive_int(values, "vocab_size", context),
        hidden_size=hidden_size,
        max_position_embeddings=_positive_int(values, "max_position_embeddings", context),
        layers=(layer,) * _positive_int(values, "num_hidden_layers", context),
        final_norm=norm,
        position=ScalarRoPESpec(_positive_float(values, "rope_theta", context)),
        output=OutputSpec(_required_bool(values, "tie_word_embeddings", context)),
        bos_token_id=_optional_token_id(values, "bos_token_id", context),
        eos_token_id=eos_token_id,
        pad_token_id=_optional_token_id(values, "pad_token_id", context),
        additional_eos_token_ids=_generation_stop_ids(model_dir, eos_token_id),
    )


def _parse_qwen3(values: dict, model_dir: Path) -> ModelSpec:
    context = "config"
    _validate_execution_options(values, context)
    _require_exact(values, "hidden_act", "silu", context)
    _require_exact(values, "attention_bias", False, context)
    query_heads = _positive_int(values, "num_attention_heads", context)
    key_value_heads = _positive_int(values, "num_key_value_heads", context)
    _validate_attention_dimensions(query_heads, key_value_heads, context)
    hidden_size = _positive_int(values, "hidden_size", context)
    head_dim = _positive_int(values, "head_dim", context)
    norm = NormSpec(NormMode.DIRECT, _positive_float(values, "rms_norm_eps", context))
    attention = SoftmaxAttentionSpec(
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        qkv_bias=False,
        qk_norm=norm,
        output_gate=False,
    )
    layer = _dense_layer(
        norm,
        attention,
        _positive_int(values, "intermediate_size", context),
    )
    eos_token_id = _token_id(values, "eos_token_id", context)
    return ModelSpec(
        architecture=Architecture.QWEN3,
        vocab_size=_positive_int(values, "vocab_size", context),
        hidden_size=hidden_size,
        max_position_embeddings=_positive_int(values, "max_position_embeddings", context),
        layers=(layer,) * _positive_int(values, "num_hidden_layers", context),
        final_norm=norm,
        position=ScalarRoPESpec(_positive_float(values, "rope_theta", context)),
        output=OutputSpec(_required_bool(values, "tie_word_embeddings", context)),
        bos_token_id=_optional_token_id(values, "bos_token_id", context),
        eos_token_id=eos_token_id,
        pad_token_id=_optional_token_id(values, "pad_token_id", context),
        additional_eos_token_ids=_generation_stop_ids(model_dir, eos_token_id),
    )


def _parse_qwen3_5(values: dict, model_dir: Path) -> ModelSpec:
    context = "config.text_config"
    text = _required(values, "text_config", "config")
    if not isinstance(text, dict):
        raise ValueError("config.text_config must be a JSON object")
    if text.get("model_type") != "qwen3_5_text":
        raise ValueError("config.text_config.model_type must be 'qwen3_5_text'")
    expert_fields = {
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    }
    present_expert_fields = sorted(expert_fields.intersection(text))
    if present_expert_fields:
        raise ValueError(
            f"config.text_config contains MoE fields outside the dense scope: {present_expert_fields}"
        )
    _validate_execution_options(text, context)
    _require_exact(text, "hidden_act", "silu", context)
    _require_exact(text, "attention_bias", False, context)
    _require_exact(text, "mamba_ssm_dtype", "float32", context)

    query_heads = _positive_int(text, "num_attention_heads", context)
    key_value_heads = _positive_int(text, "num_key_value_heads", context)
    _validate_attention_dimensions(query_heads, key_value_heads, context)
    head_dim = _positive_int(text, "head_dim", context)
    hidden_size = _positive_int(text, "hidden_size", context)
    norm = NormSpec(NormMode.UNIT_OFFSET, _positive_float(text, "rms_norm_eps", context))
    attention = SoftmaxAttentionSpec(
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        qkv_bias=False,
        qk_norm=norm,
        output_gate=_required_bool(text, "attn_output_gate", context),
    )
    delta = GatedDeltaNetSpec(
        key_heads=_positive_int(text, "linear_num_key_heads", context),
        value_heads=_positive_int(text, "linear_num_value_heads", context),
        key_head_dim=_positive_int(text, "linear_key_head_dim", context),
        value_head_dim=_positive_int(text, "linear_value_head_dim", context),
        conv_kernel=_positive_int(text, "linear_conv_kernel_dim", context),
    )
    if delta.value_heads % delta.key_heads:
        raise ValueError(
            "config.text_config.linear_num_value_heads must be divisible by linear_num_key_heads"
        )

    rope = _required(text, "rope_parameters", context)
    if not isinstance(rope, dict):
        raise ValueError("config.text_config.rope_parameters must be a JSON object")
    if rope.get("rope_type") != "default":
        raise ValueError("config.text_config.rope_parameters.rope_type must be 'default'")
    rotary_fraction = _positive_float(rope, "partial_rotary_factor", f"{context}.rope_parameters")
    if rotary_fraction > 1:
        raise ValueError("config.text_config.rope_parameters.partial_rotary_factor must be <= 1")
    rotary_dim_value = head_dim * rotary_fraction
    if not rotary_dim_value.is_integer() or int(rotary_dim_value) % 2:
        raise ValueError("Qwen3.5 rotary head dimension must be an even integer")
    rotary_dim = int(rotary_dim_value)
    sections_value = _required(rope, "mrope_section", f"{context}.rope_parameters")
    if (
        not isinstance(sections_value, list)
        or len(sections_value) != 3
        or not all(
            isinstance(section, int) and not isinstance(section, bool) and section > 0
            for section in sections_value
        )
    ):
        raise ValueError("config.text_config.rope_parameters.mrope_section must have 3 positives")
    sections = tuple(sections_value)
    if sum(sections) * 2 != rotary_dim:
        raise ValueError(
            "config.text_config.rope_parameters.mrope_section must cover the rotary dimension"
        )
    if rope.get("mrope_interleaved") is not True:
        raise ValueError("config.text_config.rope_parameters.mrope_interleaved must be true")
    position = MultiaxisRoPESpec(
        theta=_positive_float(rope, "rope_theta", f"{context}.rope_parameters"),
        rotary_fraction=rotary_fraction,
        sections=sections,
        interleaved=True,
    )

    layer_types = _required(text, "layer_types", context)
    num_layers = _positive_int(text, "num_hidden_layers", context)
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise ValueError("config.text_config.layer_types must match num_hidden_layers")
    if not all(isinstance(layer_type, str) for layer_type in layer_types):
        raise ValueError("config.text_config.layer_types must contain only strings")
    intermediate_size = _positive_int(text, "intermediate_size", context)
    token_mixers = {"full_attention": attention, "linear_attention": delta}
    unknown_layer_types = sorted(set(layer_types).difference(token_mixers))
    if unknown_layer_types:
        raise ValueError(
            f"config.text_config.layer_types contains unsupported values: {unknown_layer_types}"
        )
    layers = tuple(
        _dense_layer(norm, token_mixers[layer_type], intermediate_size)
        for layer_type in layer_types
    )
    eos_token_id = _token_id(text, "eos_token_id", context)
    return ModelSpec(
        architecture=Architecture.QWEN3_5,
        vocab_size=_positive_int(text, "vocab_size", context),
        hidden_size=hidden_size,
        max_position_embeddings=_positive_int(text, "max_position_embeddings", context),
        layers=layers,
        final_norm=norm,
        position=position,
        output=OutputSpec(_required_bool(values, "tie_word_embeddings", "config")),
        bos_token_id=_optional_token_id(text, "bos_token_id", context),
        eos_token_id=eos_token_id,
        pad_token_id=_optional_token_id(text, "pad_token_id", context),
        additional_eos_token_ids=_generation_stop_ids(model_dir, eos_token_id),
    )


_PARSERS = {
    ("qwen2", "Qwen2ForCausalLM"): _parse_qwen2,
    ("qwen3", "Qwen3ForCausalLM"): _parse_qwen3,
    ("qwen3_5", "Qwen3_5ForConditionalGeneration"): _parse_qwen3_5,
}


def load_model_spec(model_dir: str | Path) -> ModelSpec:
    model_dir = Path(model_dir)
    values = _read_object(model_dir / "config.json")
    architectures = values.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("config.architectures must contain exactly one architecture")
    architecture = architectures[0]
    if not isinstance(architecture, str):
        raise ValueError("config.architectures must contain a string architecture name")
    model_type = values.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError("config.model_type must be a string")
    key = (model_type, architecture)
    parser = _PARSERS.get(key)
    if parser is None:
        supported = ", ".join(
            f"{model_type}/{architecture}" for model_type, architecture in _PARSERS
        )
        raise ValueError(
            f"TinyInfer supported dense architectures are {supported}; received {key!r}"
        )
    return parser(values, model_dir)
