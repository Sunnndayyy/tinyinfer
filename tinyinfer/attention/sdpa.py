"""Attention backed by PyTorch scaled-dot-product attention."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal_mask: Tensor | None,
) -> Tensor:
    """Let PyTorch select the best available scaled-attention kernel."""
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=0.0,
    )
