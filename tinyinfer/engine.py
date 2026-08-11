from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from tinyinfer.architecture import Architecture
from tinyinfer.attention import AttentionName, validate_attention_name
from tinyinfer.decoding import Decoder, FinishReason, TokenEvent, create_decoder
from tinyinfer.model import QwenForCausalLM
from tinyinfer.quantization import read_quantization_config
from tinyinfer.runtime import (
    DEFAULT_ATTENTION,
    DEFAULT_DECODING,
    DEFAULT_KV_CACHE,
    DEFAULT_QUANTIZATION,
    KV_CACHE_NAMES,
    QUANTIZATION_NAMES,
)
from tinyinfer.tokenizer import Message, QwenTokenizer


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: FinishReason


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "mps" else torch.float32
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return choices[name]


class Engine:
    def __init__(
        self,
        model: QwenForCausalLM,
        tokenizer: QwenTokenizer,
        device: torch.device,
        *,
        decoding_name: str = DEFAULT_DECODING,
        attention_name: str = DEFAULT_ATTENTION,
        kv_cache_name: str = DEFAULT_KV_CACHE,
        kv_cache_block_size: int = 16,
        thinking: bool = True,
        source_model: str | None = None,
        source_revision: str | None = None,
        artifact_path: str | None = None,
    ):
        selected_attention = validate_attention_name(attention_name)
        if kv_cache_name not in KV_CACHE_NAMES:
            choices = ", ".join(KV_CACHE_NAMES)
            raise ValueError(f"unknown KV cache {kv_cache_name!r}; expected one of: {choices}")
        if kv_cache_block_size < 1:
            raise ValueError("kv_cache_block_size must be at least 1")
        if not thinking and model.config.architecture is not Architecture.QWEN3:
            raise ValueError("disabling thinking is only supported for Qwen3 models")
        self.model = model
        if self.model.attention_name != selected_attention:
            self.model.set_attention(selected_attention)
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_name = kv_cache_name
        self.kv_cache_block_size = kv_cache_block_size
        self.thinking = thinking
        self.source_model = source_model
        self.source_revision = source_revision
        self.artifact_path = artifact_path
        self.decoder: Decoder = create_decoder(
            decoding_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            kv_cache_name=kv_cache_name,
            kv_cache_block_size=kv_cache_block_size,
        )
        self.decoding_name = self.decoder.name

    @property
    def last_cache_bytes(self) -> int:
        return self.decoder.last_cache_bytes

    @property
    def attention_name(self) -> AttentionName:
        return self.model.attention_name

    @property
    def activation_dtype(self) -> torch.dtype:
        return self.model.activation_dtype

    @property
    def quantization_name(self) -> str:
        return self.model.quantization_name

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device_name: str = "auto",
        dtype_name: str = "auto",
        decoding_name: str = DEFAULT_DECODING,
        attention_name: str = DEFAULT_ATTENTION,
        kv_cache_name: str = DEFAULT_KV_CACHE,
        quantization_name: str = DEFAULT_QUANTIZATION,
        thinking: bool = True,
        model_name: str | None = None,
    ) -> Engine:
        model_path = Path(model_dir).resolve()
        quantization = read_quantization_config(model_path)
        detected_quantization = quantization.quantization if quantization else "none"
        if quantization_name != "auto" and quantization_name not in QUANTIZATION_NAMES:
            choices = ", ".join(("auto", *QUANTIZATION_NAMES))
            raise ValueError(
                f"unknown quantization {quantization_name!r}; expected one of: {choices}"
            )
        if quantization_name != "auto" and quantization_name != detected_quantization:
            raise ValueError(
                f"requested quantization {quantization_name!r} does not match "
                f"checkpoint format {detected_quantization!r}"
            )
        device = resolve_device(device_name)
        dtype = resolve_dtype(dtype_name, device)
        tokenizer = QwenTokenizer(model_path)
        model = QwenForCausalLM.from_pretrained(
            model_path,
            device=device,
            dtype=dtype,
            attention_name=attention_name,
        )
        if quantization:
            source_model = quantization.source_model
            source_revision = quantization.source_revision
        else:
            source_model = model_name or str(model_dir)
            source_revision = model_path.name if model_path.parent.name == "snapshots" else None
        return cls(
            model,
            tokenizer,
            device,
            decoding_name=decoding_name,
            attention_name=attention_name,
            kv_cache_name=kv_cache_name,
            thinking=thinking,
            source_model=source_model,
            source_revision=source_revision,
            artifact_path=str(model_path),
        )

    def stream(self, messages: list[Message], *, max_new_tokens: int) -> Iterator[TokenEvent]:
        """Validate eagerly, then return the lazy token iterator."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        prompt_ids = self.tokenizer.encode_chat(messages, thinking=self.thinking)
        if len(prompt_ids) + max_new_tokens - 1 > self.model.config.max_position_embeddings:
            raise ValueError(
                "prompt plus requested output exceeds the model's maximum sequence length"
            )
        return self.decoder.stream(prompt_ids, max_new_tokens=max_new_tokens)

    def generate(self, messages: list[Message], *, max_new_tokens: int) -> GenerationResult:
        events = self.stream(messages, max_new_tokens=max_new_tokens)
        text_parts: list[str] = []
        while True:
            try:
                text_parts.append(next(events).text)
            except StopIteration as stopped:
                finish_reason = stopped.value or "stop"
                return GenerationResult(text="".join(text_parts), finish_reason=finish_reason)
