from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from tinyinfer.attention import AttentionName, validate_attention_name
from tinyinfer.kv_cache import create_kv_cache
from tinyinfer.model import QwenForCausalLM
from tinyinfer.runtime import DEFAULT_ATTENTION, DEFAULT_KV_CACHE, KV_CACHE_NAMES
from tinyinfer.tokenizer import Message, QwenTokenizer


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    text: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: Literal["stop", "length"]


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
        attention_name: str = DEFAULT_ATTENTION,
        kv_cache_name: str = DEFAULT_KV_CACHE,
        kv_cache_block_size: int = 16,
    ):
        selected_attention = validate_attention_name(attention_name)
        if kv_cache_name not in KV_CACHE_NAMES:
            choices = ", ".join(KV_CACHE_NAMES)
            raise ValueError(f"unknown KV cache {kv_cache_name!r}; expected one of: {choices}")
        if kv_cache_block_size < 1:
            raise ValueError("kv_cache_block_size must be at least 1")
        self.model = model
        if self.model.attention_name != selected_attention:
            self.model.set_attention(selected_attention)
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_name = kv_cache_name
        self.kv_cache_block_size = kv_cache_block_size
        self.last_cache_bytes = 0

    @property
    def attention_name(self) -> AttentionName:
        return self.model.attention_name

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device_name: str = "auto",
        dtype_name: str = "auto",
        attention_name: str = DEFAULT_ATTENTION,
        kv_cache_name: str = DEFAULT_KV_CACHE,
    ) -> Engine:
        device = resolve_device(device_name)
        dtype = resolve_dtype(dtype_name, device)
        tokenizer = QwenTokenizer(model_dir)
        model = QwenForCausalLM.from_pretrained(
            model_dir,
            device=device,
            dtype=dtype,
            attention_name=attention_name,
        )
        return cls(
            model,
            tokenizer,
            device,
            attention_name=attention_name,
            kv_cache_name=kv_cache_name,
        )

    def stream(self, messages: list[Message], *, max_new_tokens: int) -> Iterator[TokenEvent]:
        """Validate eagerly, then return the lazy token iterator."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        prompt_ids = self.tokenizer.encode_chat(messages)
        if len(prompt_ids) + max_new_tokens - 1 > self.model.config.max_position_embeddings:
            raise ValueError(
                "prompt plus requested output exceeds the model's maximum sequence length"
            )
        return self._stream_tokens(prompt_ids, max_new_tokens=max_new_tokens)

    def _stream_tokens(self, prompt_ids: list[int], *, max_new_tokens: int) -> Iterator[TokenEvent]:
        token_buffer = torch.empty(
            (1, len(prompt_ids) + max_new_tokens), device=self.device, dtype=torch.long
        )
        token_buffer[0, : len(prompt_ids)] = torch.tensor(
            prompt_ids, device=self.device, dtype=torch.long
        )
        sequence_length = len(prompt_ids)
        decoder = self.tokenizer.incremental_decoder()
        cache = create_kv_cache(
            self.kv_cache_name,
            num_layers=self.model.config.num_hidden_layers,
            batch_size=1,
            num_key_value_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.config.head_dim,
            capacity=len(prompt_ids) + max_new_tokens - 1,
            block_size=self.kv_cache_block_size,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        self.last_cache_bytes = cache.bytes_allocated

        with torch.inference_mode():
            for step in range(max_new_tokens):
                position = cache.length(0)
                if position == 0:
                    model_input = token_buffer[:, :sequence_length]
                else:
                    model_input = token_buffer[:, position:sequence_length]
                logits = self.model.next_token_logits(
                    model_input,
                    cache=cache,
                    position=position,
                )
                self.last_cache_bytes = cache.bytes_allocated
                next_token = int(torch.argmax(logits[0]).item())
                if next_token in self.model.config.stop_token_ids:
                    return "stop"

                text_delta = decoder.step(next_token)
                yield TokenEvent(token_id=next_token, text=text_delta)
                token_buffer[0, sequence_length] = next_token
                sequence_length += 1
        return "length"

    def generate(self, messages: list[Message], *, max_new_tokens: int) -> GenerationResult:
        events = self.stream(messages, max_new_tokens=max_new_tokens)
        text_parts: list[str] = []
        while True:
            try:
                text_parts.append(next(events).text)
            except StopIteration as stopped:
                finish_reason = stopped.value or "stop"
                return GenerationResult(text="".join(text_parts), finish_reason=finish_reason)
