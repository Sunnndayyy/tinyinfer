import pytest
import torch

from tinyinfer.kv_cache import create_kv_cache
from tinyinfer.kv_cache.none import NoKVCache


def cache_values(start: int, tokens: int) -> torch.Tensor:
    values = torch.arange(start, start + tokens * 4, dtype=torch.float32)
    return values.view(1, 2, tokens, 2)


def build_cache(name: str, *, capacity: int = 8, block_size: int = 2):
    return create_kv_cache(
        name,
        num_layers=2,
        batch_size=1,
        num_key_value_heads=2,
        head_dim=2,
        capacity=capacity,
        block_size=block_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_no_cache_returns_the_current_keys_and_values_without_allocating() -> None:
    cache = NoKVCache()
    keys = cache_values(0, 3)
    values = cache_values(100, 3)

    stored_keys, stored_values = cache.update(0, 0, keys, values)

    assert stored_keys is keys
    assert stored_values is values
    assert cache.bytes_allocated == 0


@pytest.mark.parametrize("name", ["contiguous", "paged"])
def test_cache_appends_tokens_and_returns_the_complete_prefix(name: str) -> None:
    cache = build_cache(name)
    first_keys = cache_values(0, 3)
    first_values = cache_values(100, 3)
    next_keys = cache_values(12, 2)
    next_values = cache_values(112, 2)

    cache.update(0, 0, first_keys, first_values)
    stored_keys, stored_values = cache.update(0, 3, next_keys, next_values)

    torch.testing.assert_close(stored_keys, torch.cat((first_keys, next_keys), dim=2))
    torch.testing.assert_close(stored_values, torch.cat((first_values, next_values), dim=2))
    assert cache.length(0) == 5


@pytest.mark.parametrize("name", ["contiguous", "paged"])
def test_cache_tracks_each_transformer_layer_independently(name: str) -> None:
    cache = build_cache(name)

    cache.update(0, 0, cache_values(0, 2), cache_values(100, 2))
    cache.update(1, 0, cache_values(20, 2), cache_values(120, 2))

    assert cache.length(0) == 2
    assert cache.length(1) == 2


@pytest.mark.parametrize("name", ["contiguous", "paged"])
def test_cache_rejects_gaps_and_capacity_overflow(name: str) -> None:
    cache = build_cache(name, capacity=3)
    keys = cache_values(0, 2)
    values = cache_values(100, 2)

    with pytest.raises(ValueError, match="next cache position"):
        cache.update(0, 1, keys, values)

    cache.update(0, 0, keys, values)
    with pytest.raises(ValueError, match="capacity"):
        cache.update(0, 2, keys, values)


def test_contiguous_cache_allocates_its_capacity_up_front() -> None:
    cache = build_cache("contiguous", capacity=8)

    expected_elements = 2 * 1 * 2 * 8 * 2 * 2  # layers * batch * heads * tokens * dim * K/V
    assert (
        cache.bytes_allocated
        == expected_elements * torch.tensor([], dtype=torch.float32).element_size()
    )


def test_paged_cache_allocates_blocks_only_when_tokens_arrive() -> None:
    cache = build_cache("paged", capacity=8, block_size=2)

    assert cache.bytes_allocated == 0
    cache.update(0, 0, cache_values(0, 3), cache_values(100, 3))

    expected_elements = 2 * 1 * 2 * 2 * 2  # two blocks * batch * heads * block * dim
    expected_elements *= 2  # keys and values
    assert (
        cache.bytes_allocated
        == expected_elements * torch.tensor([], dtype=torch.float32).element_size()
    )


@pytest.mark.parametrize("name", ["contiguous", "paged"])
def test_clear_makes_a_cache_reusable(name: str) -> None:
    cache = build_cache(name)
    cache.update(0, 0, cache_values(0, 2), cache_values(100, 2))

    cache.clear()
    stored_keys, _ = cache.update(0, 0, cache_values(20, 1), cache_values(120, 1))

    assert cache.length(0) == 1
    assert stored_keys.shape[2] == 1
