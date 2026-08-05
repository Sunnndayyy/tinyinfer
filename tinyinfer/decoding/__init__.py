"""Algorithms that decide how candidate output tokens are produced."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from tinyinfer.model import QwenForCausalLM
from tinyinfer.tokenizer import QwenTokenizer

DecodingName = Literal["autoregressive"]
DECODING_NAMES: tuple[DecodingName, ...] = ("autoregressive",)
FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    text: str


class Decoder(Protocol):
    """The interface the engine needs from a decoding implementation."""

    name: DecodingName
    last_cache_bytes: int

    def stream(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
    ) -> Iterator[TokenEvent]: ...


def create_decoder(
    name: str,
    *,
    model: QwenForCausalLM,
    tokenizer: QwenTokenizer,
    device: torch.device,
    kv_cache_name: str,
    kv_cache_block_size: int,
) -> Decoder:
    """Build a decoding implementation from an explicit runtime name."""
    if name == "autoregressive":
        from tinyinfer.decoding.autoregressive import AutoregressiveDecoder

        return AutoregressiveDecoder(
            model,
            tokenizer,
            device,
            kv_cache_name=kv_cache_name,
            kv_cache_block_size=kv_cache_block_size,
        )
    choices = ", ".join(DECODING_NAMES)
    raise ValueError(f"unknown decoding implementation {name!r}; expected one of: {choices}")
