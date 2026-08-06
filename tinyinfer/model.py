from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from tinyinfer.artifacts import weight_shards
from tinyinfer.attention import (
    DEFAULT_ATTENTION,
    AttentionImplementation,
    AttentionName,
    create_attention,
)
from tinyinfer.kv_cache import KVCache
from tinyinfer.kv_cache.none import NoKVCache
from tinyinfer.quantization import Q8Embedding, Q8Linear, read_quantization_config
from tinyinfer.quantization.format import scale_name
from tinyinfer.quantization.metal import require_q8_mps


@dataclass(frozen=True)
class QwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None = None
    tie_word_embeddings: bool = True
    additional_eos_token_ids: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return frozenset((self.eos_token_id, *self.additional_eos_token_ids))

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> QwenConfig:
        values = json.loads((Path(model_dir) / "config.json").read_text())
        architectures = values.get("architectures", [])
        if values.get("model_type") != "qwen2" or "Qwen2ForCausalLM" not in architectures:
            raise ValueError(
                "TinyInfer V0 supports Qwen2ForCausalLM checkpoints only; "
                f"received model_type={values.get('model_type')!r}, architectures={architectures!r}"
            )
        if not values.get("tie_word_embeddings", False):
            raise ValueError("TinyInfer V0 expects Qwen's tied input and output embeddings")

        config_values = {
            field.name: values[field.name] for field in fields(cls) if field.name in values
        }
        generation_path = Path(model_dir) / "generation_config.json"
        if generation_path.is_file():
            generation_values = json.loads(generation_path.read_text())
            generation_eos = generation_values.get("eos_token_id", [])
            if isinstance(generation_eos, list):
                config_values["additional_eos_token_ids"] = tuple(
                    token_id for token_id in generation_eos if token_id != values["eos_token_id"]
                )
        return cls(**config_values)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


def rotate_half(values: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotary_positions(
    config: QwenConfig,
    sequence_length: int,
    device: torch.device,
    *,
    start: int = 0,
) -> tuple[Tensor, Tensor]:
    frequencies = 1.0 / (
        config.rope_theta
        ** (
            torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32)
            / config.head_dim
        )
    )
    positions = torch.arange(start, start + sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]


def apply_rotary(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.to(query.dtype)
    sin = sin.to(query.dtype)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    """Expand K/V heads so every query head has a K/V head to attend to."""
    if repeats == 1:
        return hidden_states
    batch, key_value_heads, sequence_length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repeats, sequence_length, head_dim
    )
    return expanded.reshape(batch, key_value_heads * repeats, sequence_length, head_dim)


def causal_attention_mask(
    query_length: int,
    key_length: int,
    query_start: int,
    device: torch.device,
) -> Tensor | None:
    """Allow each query to see past keys and earlier keys in its own chunk."""
    if query_length == 1:
        return None
    query_positions = torch.arange(
        query_start, query_start + query_length, device=device
    ).unsqueeze(1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


class Attention(nn.Module):
    def __init__(self, config: QwenConfig, implementation: AttentionImplementation):
        super().__init__()
        self.config = config
        self.calculate_attention = implementation
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        causal_mask: Tensor | None,
        *,
        cache: KVCache,
        layer_index: int,
        position: int,
    ) -> Tensor:
        batch, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch, sequence_length, self.config.num_attention_heads, self.config.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch, sequence_length, self.config.num_key_value_heads, self.config.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch, sequence_length, self.config.num_key_value_heads, self.config.head_dim)
            .transpose(1, 2)
        )

        query, key = apply_rotary(query, key, cos, sin)
        key, value = cache.update(layer_index, position, key, value)
        key = repeat_kv(key, self.config.num_key_value_groups)
        value = repeat_kv(value, self.config.num_key_value_groups)

        attended = self.calculate_attention(query, key, value, causal_mask)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence_length, -1)
        return self.o_proj(attended)


class MLP(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig, attention: AttentionImplementation):
        super().__init__()
        self.self_attn = Attention(config, attention)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        causal_mask: Tensor | None,
        *,
        cache: KVCache,
        layer_index: int,
        position: int,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            causal_mask,
            cache=cache,
            layer_index=layer_index,
            position=position,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig, *, attention_name: AttentionName = DEFAULT_ATTENTION):
        super().__init__()
        self.config = config
        self.attention_name = attention_name
        attention = create_attention(attention_name)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            DecoderLayer(config, attention) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def set_attention(self, name: AttentionName) -> None:
        attention = create_attention(name)
        for layer in self.layers:
            layer.self_attn.calculate_attention = attention
        self.attention_name = name

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: KVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        if cache is None:
            cache = NoKVCache()
        hidden_states = self.embed_tokens(input_ids)
        query_length = input_ids.shape[1]
        key_length = position + query_length
        cos, sin = rotary_positions(
            self.config,
            query_length,
            input_ids.device,
            start=position,
        )
        causal_mask = causal_attention_mask(
            query_length,
            key_length,
            position,
            input_ids.device,
        )
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                causal_mask,
                cache=cache,
                layer_index=layer_index,
                position=position,
            )
        return self.norm(hidden_states)


class QwenForCausalLM(nn.Module):
    def __init__(self, config: QwenConfig, *, attention_name: AttentionName = DEFAULT_ATTENTION):
        super().__init__()
        self.config = config
        self.model = QwenModel(config, attention_name=attention_name)

    @property
    def attention_name(self) -> AttentionName:
        return self.model.attention_name

    def set_attention(self, name: AttentionName) -> None:
        self.model.set_attention(name)

    @property
    def activation_dtype(self) -> torch.dtype:
        return self.model.norm.weight.dtype

    @property
    def quantization_name(self) -> str:
        return "q8" if isinstance(self.model.embed_tokens, Q8Embedding) else "none"

    def _replace_q8_modules_(self, linear_factory, embedding_factory) -> None:
        for layer in self.model.layers:
            attention = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attention, name, linear_factory(getattr(attention, name)))
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, linear_factory(getattr(layer.mlp, name)))
        self.model.embed_tokens = embedding_factory(self.model.embed_tokens)

    def quantize_q8_(self) -> QwenForCausalLM:
        """Replace floating-point projection and embedding modules in place."""
        self._replace_q8_modules_(Q8Linear.from_float, Q8Embedding.from_float)
        return self

    def _project_logits(self, hidden_states: Tensor) -> Tensor:
        embedding = self.model.embed_tokens
        if isinstance(embedding, Q8Embedding):
            return embedding.project(hidden_states)
        return F.linear(hidden_states, embedding.weight)

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: KVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        hidden_states = self.model(input_ids, cache=cache, position=position)
        return self._project_logits(hidden_states)

    def next_token_logits(
        self,
        input_ids: Tensor,
        *,
        cache: KVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        """Project only the final position because generation ignores earlier logits."""
        final_hidden_state = self.model(input_ids, cache=cache, position=position)[:, -1, :]
        return self._project_logits(final_hidden_state)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        attention_name: AttentionName = DEFAULT_ATTENTION,
    ) -> QwenForCausalLM:
        quantization = read_quantization_config(model_dir)
        if quantization and device.type == "mps":
            require_q8_mps()
            if dtype != torch.bfloat16:
                raise ValueError("Q8 on MPS requires bfloat16 activations")
        elif quantization and device.type != "cpu":
            raise ValueError(f"Q8 checkpoints do not support device {device.type!r}")
        config = QwenConfig.from_directory(model_dir)
        with torch.device("meta"):
            model = cls(config, attention_name=attention_name)
            if quantization:
                model._replace_q8_modules_(
                    Q8Linear.empty_like,
                    lambda embedding: Q8Embedding.empty_like(embedding, dtype),
                )

        expected = set(model.state_dict())
        if quantization:
            expected_q8 = {name for name in expected if scale_name(name) in expected}
            if set(quantization.tensors) != expected_q8:
                raise ValueError("quantization metadata does not match the Qwen model")
        loaded: set[str] = set()
        unexpected: set[str] = set()
        q8_weights = set(quantization.tensors) if quantization else set()
        q8_scales = {scale_name(name) for name in q8_weights}
        for shard_path in weight_shards(model_dir):
            shard = load_file(str(shard_path), device="cpu")
            repeated = loaded.intersection(shard)
            if repeated:
                raise ValueError(f"checkpoint repeats tensors across shards: {sorted(repeated)}")
            if quantization:
                for name, tensor in shard.items():
                    expected_dtype = (
                        torch.int8
                        if name in q8_weights
                        else torch.float16
                        if name in q8_scales
                        else None
                    )
                    if expected_dtype and tensor.dtype != expected_dtype:
                        raise ValueError(
                            f"quantized tensor {name} must be {expected_dtype}, got {tensor.dtype}"
                        )
                # Packed integers and FP16 scales keep their storage dtypes.
                shard = {
                    name: tensor
                    if not tensor.is_floating_point() or name.endswith(".scales")
                    else tensor.to(dtype)
                    for name, tensor in shard.items()
                }
            result = model.load_state_dict(shard, strict=False, assign=True)
            loaded.update(shard)
            unexpected.update(result.unexpected_keys)

        if not quantization:
            unexpected.discard("lm_head.weight")
        missing = expected - loaded
        if missing or unexpected:
            raise ValueError(
                f"checkpoint does not match TinyInfer's Qwen2 implementation: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

        if quantization:
            model = model.to(device=device)
        else:
            model = model.to(device=device, dtype=dtype)
        model.eval()
        return model
