"""Readable reference attention using explicit matrix operations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def eager_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal_mask: Tensor | None,
) -> Tensor:
    """Calculate scaled attention explicitly for learning and comparison."""
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return probabilities @ value
