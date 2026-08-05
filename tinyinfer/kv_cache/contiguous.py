"""A preallocated contiguous KV cache for incremental decoding.

This is an implementation of the KV-cache idea discussed in
https://arxiv.org/abs/1911.02150 ("Fast Transformer Decoding: One Write-Head
is All You Need").

The paper motivates caching and reducing repeated K/V memory traffic. This file
implements the cache itself; Qwen's grouped-query attention determines how many
K/V heads are stored.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tinyinfer.kv_cache import validate_update


class ContiguousKVCache:
    name = "contiguous"

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_key_value_heads: int,
        head_dim: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        dimensions = (num_layers, batch_size, num_key_value_heads, capacity, head_dim)
        if any(size < 1 for size in dimensions):
            raise ValueError("contiguous cache dimensions must all be positive")
        self.capacity = capacity
        self._keys = torch.empty(dimensions, device=device, dtype=dtype)
        self._values = torch.empty(dimensions, device=device, dtype=dtype)
        self._lengths = [0] * num_layers

    @property
    def bytes_allocated(self) -> int:
        return self._keys.nbytes + self._values.nbytes

    def length(self, layer_index: int) -> int:
        return self._lengths[layer_index]

    def update(
        self,
        layer_index: int,
        position: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        validate_update(position, key, value)
        expected = self._keys[layer_index].shape
        if key.shape[:2] != expected[:2] or key.shape[3] != expected[3]:
            raise ValueError(
                "cache update shape does not match its configured batch, heads, and head dim"
            )
        if position != self._lengths[layer_index]:
            raise ValueError(
                f"next cache position for layer {layer_index} is "
                f"{self._lengths[layer_index]}, not {position}"
            )
        end = position + key.shape[2]
        if end > self.capacity:
            raise ValueError(f"cache update ends at {end}, beyond capacity {self.capacity}")

        self._keys[layer_index, :, :, position:end, :].copy_(key)
        self._values[layer_index, :, :, position:end, :].copy_(value)
        self._lengths[layer_index] = end
        return (
            self._keys[layer_index, :, :, :end, :],
            self._values[layer_index, :, :, :end, :],
        )

    def clear(self) -> None:
        self._lengths = [0] * len(self._lengths)
