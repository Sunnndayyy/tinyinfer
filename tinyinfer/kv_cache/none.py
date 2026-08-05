"""Uncached reference Transformer execution.

This is a reference implementation of https://arxiv.org/abs/1706.03762
("Attention Is All You Need").

This deliberately recomputes the complete prefix. It is the readable
correctness baseline that every cached implementation must match.
"""

from torch import Tensor

from tinyinfer.kv_cache import validate_update


class NoKVCache:
    name = "none"

    @property
    def bytes_allocated(self) -> int:
        return 0

    def length(self, layer_index: int) -> int:
        return 0

    def update(
        self,
        layer_index: int,
        position: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        validate_update(position, key, value)
        if position != 0:
            raise ValueError("uncached attention expects the complete prefix at position 0")
        return key, value

    def clear(self) -> None:
        pass
