import json

import pytest
import torch
from safetensors.torch import save_file

from tinyinfer.model import QwenConfig, QwenForCausalLM, repeat_kv


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


def test_repeat_kv_expands_grouped_query_heads() -> None:
    keys = torch.tensor([[[[1.0]], [[2.0]]]])

    repeated = repeat_kv(keys, repeats=2)

    assert repeated.shape == (1, 4, 1, 1)
    assert repeated[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_qwen_forward_returns_one_logit_vector_per_token() -> None:
    model = QwenForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 5, 7, 9]])

    logits = model(input_ids)

    assert logits.shape == (1, 4, 32)
    assert torch.isfinite(logits).all()


def test_future_tokens_do_not_change_earlier_logits() -> None:
    torch.manual_seed(7)
    model = QwenForCausalLM(tiny_config()).eval()
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
