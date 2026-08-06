"""Reference decoding that produces one token per target-model step."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from tinyinfer.decoding import DecodingName, TokenEvent
from tinyinfer.kv_cache import create_kv_cache
from tinyinfer.model import QwenForCausalLM
from tinyinfer.tokenizer import QwenTokenizer


class AutoregressiveDecoder:
    """Run readable, one-token-at-a-time greedy decoding for one model."""

    name: DecodingName = "autoregressive"

    def __init__(
        self,
        model: QwenForCausalLM,
        tokenizer: QwenTokenizer,
        device: torch.device,
        *,
        kv_cache_name: str,
        kv_cache_block_size: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_name = kv_cache_name
        self.kv_cache_block_size = kv_cache_block_size
        self.last_cache_bytes = 0

    def stream(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
    ) -> Iterator[TokenEvent]:
        token_buffer = torch.empty(
            (1, len(prompt_ids) + max_new_tokens), device=self.device, dtype=torch.long
        )
        token_buffer[0, : len(prompt_ids)] = torch.tensor(
            prompt_ids, device=self.device, dtype=torch.long
        )
        sequence_length = len(prompt_ids)
        text_decoder = self.tokenizer.incremental_decoder()
        cache = create_kv_cache(
            self.kv_cache_name,
            num_layers=self.model.config.num_hidden_layers,
            batch_size=1,
            num_key_value_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.config.head_dim,
            capacity=len(prompt_ids) + max_new_tokens - 1,
            block_size=self.kv_cache_block_size,
            device=self.device,
            dtype=self.model.activation_dtype,
        )
        self.last_cache_bytes = cache.bytes_allocated

        with torch.inference_mode():
            for _ in range(max_new_tokens):
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

                text_delta = text_decoder.step(next_token)
                yield TokenEvent(token_id=next_token, text=text_delta)
                token_buffer[0, sequence_length] = next_token
                sequence_length += 1
        return "length"
