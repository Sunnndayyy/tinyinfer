import pytest
import torch

from tinyinfer.attention import ATTENTION_NAMES, create_attention
from tinyinfer.attention.eager import eager_attention


def test_attention_names_keep_eager_as_the_reference_default_order() -> None:
    assert ATTENTION_NAMES == ("eager", "sdpa")


def test_create_attention_returns_the_eager_implementation() -> None:
    assert create_attention("eager") is eager_attention


def test_create_attention_rejects_an_unknown_implementation() -> None:
    with pytest.raises(ValueError, match="unknown attention 'flash'; expected one of: eager, sdpa"):
        create_attention("flash")


@pytest.mark.parametrize(
    ("query_length", "key_length", "query_start"),
    [
        (4, 4, 0),
        (2, 5, 3),
        (1, 5, 4),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_sdpa_matches_eager_attention(
    query_length: int,
    key_length: int,
    query_start: int,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(23)
    query = torch.randn(2, 4, query_length, 8, dtype=dtype)
    key = torch.randn(2, 4, key_length, 8, dtype=dtype)
    value = torch.randn(2, 4, key_length, 8, dtype=dtype)
    query_positions = torch.arange(query_start, query_start + query_length).unsqueeze(1)
    key_positions = torch.arange(key_length).unsqueeze(0)
    causal_mask = None if query_length == 1 else key_positions <= query_positions

    eager = create_attention("eager")(query, key, value, causal_mask)
    sdpa = create_attention("sdpa")(query, key, value, causal_mask)

    torch.testing.assert_close(sdpa, eager, rtol=rtol, atol=atol)
