"""Small, explicit attention implementations."""

from __future__ import annotations

from typing import Literal, Protocol, cast

from torch import Tensor

AttentionName = Literal["eager", "sdpa"]
ATTENTION_NAMES: tuple[AttentionName, ...] = ("eager", "sdpa")
DEFAULT_ATTENTION: AttentionName = "eager"


class AttentionImplementation(Protocol):
    """The tensor calculation shared by each model attention route."""

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        causal_mask: Tensor | None,
    ) -> Tensor: ...


def validate_attention_name(name: str) -> AttentionName:
    """Validate a runtime name and narrow it for internal APIs."""
    if name not in ATTENTION_NAMES:
        choices = ", ".join(ATTENTION_NAMES)
        raise ValueError(f"unknown attention {name!r}; expected one of: {choices}")
    return cast(AttentionName, name)


def create_attention(name: str) -> AttentionImplementation:
    """Select one attention calculation by its explicit runtime name."""
    name = validate_attention_name(name)
    if name == "eager":
        from tinyinfer.attention.eager import eager_attention

        return eager_attention
    if name == "sdpa":
        from tinyinfer.attention.sdpa import sdpa_attention

        return sdpa_attention
    raise AssertionError(f"unhandled attention implementation: {name}")


__all__ = [
    "ATTENTION_NAMES",
    "DEFAULT_ATTENTION",
    "AttentionImplementation",
    "AttentionName",
    "create_attention",
    "validate_attention_name",
]
