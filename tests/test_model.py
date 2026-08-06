import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tinyinfer.kv_cache import create_kv_cache
from tinyinfer.model import QwenConfig, QwenForCausalLM, causal_attention_mask, repeat_kv
from tinyinfer.quantization import Q8Embedding, Q8Linear, dequantize_q8, quantize_q8


def tiny_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        bos_token_id=1,
        eos_token_id=2,
    )


def q8_tiny_config() -> QwenConfig:
    return replace(tiny_config(), hidden_size=32, intermediate_size=64)


def dequantize_model_weights(model: QwenForCausalLM) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            quantized, scales = quantize_q8(module.weight)
            module.weight.data.copy_(dequantize_q8(quantized, scales))


def test_q8_model_matches_the_same_weights_explicitly_dequantized() -> None:
    torch.manual_seed(23)
    source = QwenForCausalLM(q8_tiny_config()).eval()
    expected = deepcopy(source)
    actual = deepcopy(source).quantize_q8_()
    input_ids = torch.tensor([[1, 5, 7, 9]])
    dequantize_model_weights(expected)

    with torch.inference_mode():
        expected_logits = expected(input_ids)
        actual_logits = actual(input_ids)

    assert isinstance(actual.model.embed_tokens, Q8Embedding)
    assert sum(isinstance(module, Q8Linear) for module in actual.modules()) == 14
    assert not any(isinstance(module, nn.Linear) for module in actual.modules())
    torch.testing.assert_close(actual_logits, expected_logits)


def test_repeat_kv_expands_grouped_query_heads() -> None:
    keys = torch.tensor([[[[1.0]], [[2.0]]]])

    repeated = repeat_kv(keys, repeats=2)

    assert repeated.shape == (1, 4, 1, 1)
    assert repeated[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_single_token_decode_does_not_need_a_causal_mask() -> None:
    mask = causal_attention_mask(1, 5, 4, torch.device("cpu"))

    assert mask is None


@pytest.mark.parametrize("attention_name", ["eager", "sdpa"])
def test_qwen_forward_returns_one_logit_vector_per_token(attention_name: str) -> None:
    model = QwenForCausalLM(tiny_config(), attention_name=attention_name)
    input_ids = torch.tensor([[1, 5, 7, 9]])

    logits = model(input_ids)

    assert logits.shape == (1, 4, 32)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("attention_name", ["eager", "sdpa"])
def test_future_tokens_do_not_change_earlier_logits(attention_name: str) -> None:
    torch.manual_seed(7)
    model = QwenForCausalLM(tiny_config(), attention_name=attention_name).eval()
    prefix = torch.tensor([[1, 5, 7]])
    longer = torch.tensor([[1, 5, 7, 11]])

    with torch.inference_mode():
        prefix_logits = model(prefix)
        longer_logits = model(longer)

    torch.testing.assert_close(prefix_logits, longer_logits[:, :3], rtol=1e-5, atol=1e-5)


def test_different_prefixes_change_the_final_token_prediction() -> None:
    torch.manual_seed(11)
    model = QwenForCausalLM(tiny_config()).eval()

    with torch.inference_mode():
        first = model(torch.tensor([[1, 5, 9]]))[:, -1]
        second = model(torch.tensor([[1, 6, 9]]))[:, -1]

    assert not torch.allclose(first, second)


@pytest.mark.parametrize("attention_name", ["eager", "sdpa"])
@pytest.mark.parametrize("cache_name", ["contiguous", "paged"])
def test_incremental_kv_cache_matches_full_prefix_logits(
    cache_name: str,
    attention_name: str,
) -> None:
    torch.manual_seed(13)
    model = QwenForCausalLM(tiny_config(), attention_name=attention_name).eval()
    prompt = torch.tensor([[1, 5, 7, 9]])
    cache = create_kv_cache(
        cache_name,
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        capacity=8,
        block_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        prefill_logits = model.next_token_logits(prompt, cache=cache, position=0)
        next_token = torch.argmax(prefill_logits, dim=-1, keepdim=True)
        cached_logits = model.next_token_logits(
            next_token,
            cache=cache,
            position=prompt.shape[1],
        )
        full_prefix_logits = model.next_token_logits(torch.cat((prompt, next_token), dim=1))

    torch.testing.assert_close(cached_logits, full_prefix_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("attention_name", ["eager", "sdpa"])
@pytest.mark.parametrize("cache_name", ["contiguous", "paged"])
def test_chunked_prefill_matches_full_prefix_logits(
    cache_name: str,
    attention_name: str,
) -> None:
    torch.manual_seed(17)
    model = QwenForCausalLM(tiny_config(), attention_name=attention_name).eval()
    prompt = torch.tensor([[1, 5, 7, 9]])
    cache = create_kv_cache(
        cache_name,
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        capacity=8,
        block_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        model.next_token_logits(prompt[:, :2], cache=cache, position=0)
        chunked_logits = model.next_token_logits(prompt[:, 2:], cache=cache, position=2)
        full_prefix_logits = model.next_token_logits(prompt)

    torch.testing.assert_close(chunked_logits, full_prefix_logits, rtol=1e-5, atol=1e-5)


def write_tiny_checkpoint(model, directory, *, omit_key: str | None = None) -> None:
    config = model.config
    values = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        **config.__dict__,
    }
    values.pop("additional_eos_token_ids")
    directory.joinpath("config.json").write_text(json.dumps(values))
    directory.joinpath("generation_config.json").write_text(
        json.dumps({"eos_token_id": [config.eos_token_id]})
    )

    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if omit_key:
        state.pop(omit_key)
    keys = sorted(state)
    midpoint = len(keys) // 2
    save_file({key: state[key] for key in keys[:midpoint]}, directory / "model-00001.safetensors")
    save_file({key: state[key] for key in keys[midpoint:]}, directory / "model-00002.safetensors")


def test_multishard_checkpoint_loader_reproduces_tiny_model(tmp_path) -> None:
    torch.manual_seed(19)
    source = QwenForCausalLM(tiny_config()).eval()
    write_tiny_checkpoint(source, tmp_path)

    loaded = QwenForCausalLM.from_pretrained(
        tmp_path, device=torch.device("cpu"), dtype=torch.float32
    )

    assert loaded.training is False
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)


def test_checkpoint_loader_rejects_a_missing_tensor(tmp_path) -> None:
    source = QwenForCausalLM(tiny_config())
    missing_key = "model.norm.weight"
    write_tiny_checkpoint(source, tmp_path, omit_key=missing_key)

    with pytest.raises(ValueError, match=missing_key):
        QwenForCausalLM.from_pretrained(tmp_path, device=torch.device("cpu"), dtype=torch.float32)
