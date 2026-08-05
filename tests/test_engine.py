from types import SimpleNamespace

import pytest
import torch

from tinyinfer.engine import Engine
from tinyinfer.tokenizer import Message


class Decoder:
    def step(self, token_id: int) -> str:
        return f"<{token_id}>"


class Tokenizer:
    def encode_chat(self, messages) -> list[int]:
        return [1, 2, 3]

    def incremental_decoder(self) -> Decoder:
        return Decoder()


class RecordingModel:
    def __init__(self) -> None:
        self.calls = []
        self.cache_dtypes = []
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=2,
            max_position_embeddings=16,
            stop_token_ids=frozenset(),
        )

    def parameters(self):
        return iter((self.weight,))

    def next_token_logits(self, input_ids, *, cache, position):
        self.calls.append((input_ids.shape[1], position, type(cache).__name__))
        keys = torch.zeros(
            (1, 1, input_ids.shape[1], 2),
            dtype=self.weight.dtype,
        )
        stored_keys, _ = cache.update(0, position, keys, keys)
        self.cache_dtypes.append(stored_keys.dtype)
        logits = torch.zeros((1, 8))
        logits[0, 4] = 1
        return logits


@pytest.mark.parametrize(
    ("cache_name", "expected_lengths", "expected_positions"),
    [
        ("none", [3, 4, 5], [0, 0, 0]),
        ("contiguous", [3, 1, 1], [0, 3, 4]),
        ("paged", [3, 1, 1], [0, 3, 4]),
    ],
)
def test_engine_selects_full_prefix_or_incremental_decode(
    cache_name: str,
    expected_lengths: list[int],
    expected_positions: list[int],
) -> None:
    model = RecordingModel()
    engine = Engine(model, Tokenizer(), torch.device("cpu"), kv_cache_name=cache_name)

    events = list(engine.stream([Message(role="user", content="hello")], max_new_tokens=3))

    assert [event.token_id for event in events] == [4, 4, 4]
    assert [call[0] for call in model.calls] == expected_lengths
    assert [call[1] for call in model.calls] == expected_positions


def test_engine_uses_the_models_dtype_for_cache_storage() -> None:
    model = RecordingModel()
    model.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    engine = Engine(model, Tokenizer(), torch.device("cpu"), kv_cache_name="contiguous")

    list(engine.stream([Message(role="user", content="hello")], max_new_tokens=1))

    assert model.cache_dtypes == [torch.bfloat16]
