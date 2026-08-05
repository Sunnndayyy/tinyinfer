import json

import pytest

from tinyinfer.model import QwenConfig


def test_config_loads_supported_qwen2_architecture(tmp_path) -> None:
    values = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(values))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}))

    config = QwenConfig.from_directory(tmp_path)

    assert config.hidden_size == 16
    assert config.head_dim == 4
    assert config.num_key_value_groups == 2
    assert config.stop_token_ids == frozenset({2, 3})


def test_config_rejects_a_different_model_architecture(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
    )

    with pytest.raises(ValueError, match="Qwen2"):
        QwenConfig.from_directory(tmp_path)
