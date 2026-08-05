"""This is an implementation of the block-allocation idea in
https://arxiv.org/abs/2309.06180 ("Efficient Memory Management for Large
Language Model Serving with PagedAttention").

TODO for full PagedAttention:
- allocate physical K/V blocks from a pool shared by concurrent requests;
- give each sequence a logical-to-physical block table;
- make attention read those blocks directly without ``torch.cat``;
- let the scheduler allocate, share, copy-on-write, and free blocks safely;
- benchmark fragmentation, memory capacity, throughput, and output parity.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tinyinfer.kv_cache import validate_update


class PagedKVCache:
    name = "paged"

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_key_value_heads: int,
        head_dim: int,
        capacity: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        dimensions = (num_layers, batch_size, num_key_value_heads, head_dim, capacity, block_size)
        if any(size < 1 for size in dimensions):
            raise ValueError("paged cache dimensions must all be positive")
        self.capacity = capacity
        self.block_size = block_size
        self._block_shape = (batch_size, num_key_value_heads, block_size, head_dim)
        self._device = device
        self._dtype = dtype
        self._key_blocks: list[list[Tensor]] = [[] for _ in range(num_layers)]
        self._value_blocks: list[list[Tensor]] = [[] for _ in range(num_layers)]
        self._lengths = [0] * num_layers
        self._bytes_allocated = 0

    @property
    def bytes_allocated(self) -> int:
        return self._bytes_allocated

    def length(self, layer_index: int) -> int:
        return self._lengths[layer_index]

    def _allocate_block(self, layer_index: int) -> None:
        key = torch.empty(self._block_shape, device=self._device, dtype=self._dtype)
        value = torch.empty(self._block_shape, device=self._device, dtype=self._dtype)
        self._key_blocks[layer_index].append(key)
        self._value_blocks[layer_index].append(value)
        self._bytes_allocated += key.nbytes + value.nbytes

    def update(
        self,
        layer_index: int,
        position: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        validate_update(position, key, value)
        if key.shape[:2] != self._block_shape[:2] or key.shape[3] != self._block_shape[3]:
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

        source_position = 0
        while position < end:
            block_index, block_position = divmod(position, self.block_size)
            if block_index == len(self._key_blocks[layer_index]):
                self._allocate_block(layer_index)
            tokens = min(self.block_size - block_position, end - position)
            source_end = source_position + tokens
            self._key_blocks[layer_index][block_index][
                :, :, block_position : block_position + tokens, :
            ].copy_(key[:, :, source_position:source_end, :])
            self._value_blocks[layer_index][block_index][
                :, :, block_position : block_position + tokens, :
            ].copy_(value[:, :, source_position:source_end, :])
            position += tokens
            source_position = source_end

        self._lengths[layer_index] = end
        return self._materialize(self._key_blocks[layer_index], end), self._materialize(
            self._value_blocks[layer_index], end
        )

    def _materialize(self, blocks: list[Tensor], length: int) -> Tensor:
        chunks = []
        remaining = length
        for block in blocks:
            tokens = min(self.block_size, remaining)
            chunks.append(block[:, :, :tokens, :])
            remaining -= tokens
            if remaining == 0:
                break
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)

    def clear(self) -> None:
        for blocks in self._key_blocks:
            blocks.clear()
        for blocks in self._value_blocks:
            blocks.clear()
        self._lengths = [0] * len(self._lengths)
        self._bytes_allocated = 0
