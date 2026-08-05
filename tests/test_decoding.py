import pytest
import torch

from tinyinfer.decoding import DECODING_NAMES, create_decoder
from tinyinfer.decoding.autoregressive import AutoregressiveDecoder


def test_create_decoder_builds_the_selected_implementation() -> None:
    decoder = create_decoder(
        "autoregressive",
        model=object(),
        tokenizer=object(),
        device=torch.device("cpu"),
        kv_cache_name="contiguous",
        kv_cache_block_size=16,
    )

    assert DECODING_NAMES == ("autoregressive",)
    assert isinstance(decoder, AutoregressiveDecoder)
    assert decoder.name == "autoregressive"


def test_create_decoder_rejects_an_unknown_implementation() -> None:
    with pytest.raises(ValueError, match="unknown decoding implementation 'speculative'"):
        create_decoder(
            "speculative",
            model=object(),
            tokenizer=object(),
            device=torch.device("cpu"),
            kv_cache_name="contiguous",
            kv_cache_block_size=16,
        )
