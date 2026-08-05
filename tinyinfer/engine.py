from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from tinyinfer.model import QwenForCausalLM
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
    def __init__(self, model: QwenForCausalLM, tokenizer: QwenTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device_name: str = "auto",
        dtype_name: str = "auto",
    ) -> Engine:
        device = resolve_device(device_name)
        dtype = resolve_dtype(dtype_name, device)
        tokenizer = QwenTokenizer(model_dir)
        model = QwenForCausalLM.from_pretrained(model_dir, device=device, dtype=dtype)
        return cls(model, tokenizer, device)

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

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = self.model.next_token_logits(token_buffer[:, :sequence_length])
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
