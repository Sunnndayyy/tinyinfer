"""Small, explicit KV-cache implementations."""

from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch import Tensor

KVCacheName = Literal["none", "contiguous", "paged"]
KV_CACHE_NAMES: tuple[KVCacheName, ...] = ("none", "contiguous", "paged")


class KVCache(Protocol):
    """The smallest interface attention needs from a KV cache."""

    name: KVCacheName

    @property
    def bytes_allocated(self) -> int: ...

    def length(self, layer_index: int) -> int: ...

    def update(
        self,
        layer_index: int,
        position: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def clear(self) -> None: ...


def validate_update(position: int, key: Tensor, value: Tensor) -> None:
    if position < 0:
        raise ValueError("cache position must be non-negative")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("cache keys and values must have shape [batch, heads, tokens, dim]")
    if key.shape != value.shape:
        raise ValueError("cache keys and values must have matching shapes")
    if key.shape[2] < 1:
        raise ValueError("a cache update must contain at least one token")


def create_kv_cache(
    name: str,
    *,
    num_layers: int,
    batch_size: int,
    num_key_value_heads: int,
    head_dim: int,
    capacity: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVCache:
    """Build one request-local cache from an explicit runtime name."""
    if name == "none":
        from tinyinfer.kv_cache.none import NoKVCache

        return NoKVCache()

    arguments = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "capacity": capacity,
        "device": device,
        "dtype": dtype,
    }
    if name == "contiguous":
        from tinyinfer.kv_cache.contiguous import ContiguousKVCache

        return ContiguousKVCache(**arguments)
    if name == "paged":
        from tinyinfer.kv_cache.paged import PagedKVCache

        return PagedKVCache(**arguments, block_size=block_size)
    choices = ", ".join(KV_CACHE_NAMES)
    raise ValueError(f"unknown KV cache {name!r}; expected one of: {choices}")
