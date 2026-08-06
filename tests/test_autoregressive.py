from types import SimpleNamespace

import pytest
import torch

from tinyinfer.decoding.autoregressive import AutoregressiveDecoder


class Decoder:
    def step(self, token_id: int) -> str:
        return f"<{token_id}>"


class Tokenizer:
    def incremental_decoder(self) -> Decoder:
        return Decoder()


class RecordingModel:
    def __init__(self, token_ids: list[int] | None = None) -> None:
        self.calls = []
        self.cache_dtypes = []
        self.token_ids = iter(token_ids or [4, 4, 4])
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=2,
            stop_token_ids=frozenset({7}),
        )

    def parameters(self):
        return iter((self.weight,))

    @property
    def activation_dtype(self) -> torch.dtype:
        return self.weight.dtype

    def next_token_logits(self, input_ids, *, cache, position):
        self.calls.append((input_ids.shape[1], position, type(cache).__name__))
        keys = torch.zeros(
            (1, 1, input_ids.shape[1], 2),
            dtype=self.weight.dtype,
        )
        stored_keys, _ = cache.update(0, position, keys, keys)
        self.cache_dtypes.append(stored_keys.dtype)
        logits = torch.zeros((1, 8))
        logits[0, next(self.token_ids)] = 1
        return logits


@pytest.mark.parametrize(
    ("cache_name", "expected_lengths", "expected_positions"),
    [
        ("none", [3, 4, 5], [0, 0, 0]),
        ("contiguous", [3, 1, 1], [0, 3, 4]),
        ("paged", [3, 1, 1], [0, 3, 4]),
    ],
)
def test_autoregressive_decoder_selects_full_prefix_or_incremental_decode(
    cache_name: str,
    expected_lengths: list[int],
    expected_positions: list[int],
) -> None:
    model = RecordingModel()
    decoder = AutoregressiveDecoder(
        model,
        Tokenizer(),
        torch.device("cpu"),
        kv_cache_name=cache_name,
        kv_cache_block_size=16,
    )

    events = list(decoder.stream([1, 2, 3], max_new_tokens=3))

    assert [event.token_id for event in events] == [4, 4, 4]
    assert [event.text for event in events] == ["<4>", "<4>", "<4>"]
    assert [call[0] for call in model.calls] == expected_lengths
    assert [call[1] for call in model.calls] == expected_positions


def test_autoregressive_decoder_stops_before_emitting_a_stop_token() -> None:
    decoder = AutoregressiveDecoder(
        RecordingModel([4, 7]),
        Tokenizer(),
        torch.device("cpu"),
        kv_cache_name="contiguous",
        kv_cache_block_size=16,
    )

    events = decoder.stream([1, 2, 3], max_new_tokens=3)

    assert next(events).token_id == 4
    with pytest.raises(StopIteration) as stopped:
        next(events)
    assert stopped.value.value == "stop"


def test_autoregressive_decoder_reports_the_length_limit() -> None:
    decoder = AutoregressiveDecoder(
        RecordingModel([4]),
        Tokenizer(),
        torch.device("cpu"),
        kv_cache_name="contiguous",
        kv_cache_block_size=16,
    )

    events = decoder.stream([1, 2, 3], max_new_tokens=1)

    assert next(events).token_id == 4
    with pytest.raises(StopIteration) as stopped:
        next(events)
    assert stopped.value.value == "length"


def test_autoregressive_decoder_uses_the_models_dtype_for_cache_storage() -> None:
    model = RecordingModel()
    model.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    decoder = AutoregressiveDecoder(
        model,
        Tokenizer(),
        torch.device("cpu"),
        kv_cache_name="contiguous",
        kv_cache_block_size=16,
    )

    list(decoder.stream([1, 2, 3], max_new_tokens=1))

    assert model.cache_dtypes == [torch.bfloat16]
    assert decoder.last_cache_bytes > 0
