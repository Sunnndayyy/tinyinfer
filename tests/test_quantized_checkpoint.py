import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinyinfer.kv_cache import create_kv_cache
from tinyinfer.model import QwenConfig, QwenForCausalLM
from tinyinfer.quantization import Q8Embedding, Q8Linear
from tinyinfer.quantization import convert as convert_module
from tinyinfer.quantization.convert import convert_checkpoint
from tinyinfer.quantization.format import QuantizationConfig


def tiny_q8_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def write_checkpoint(model: QwenForCausalLM, directory: Path) -> None:
    directory.mkdir()
    values = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        **model.config.__dict__,
    }
    values.pop("additional_eos_token_ids")
    directory.joinpath("config.json").write_text(json.dumps(values))
    directory.joinpath("generation_config.json").write_text(
        json.dumps({"eos_token_id": [model.config.eos_token_id]})
    )
    directory.joinpath("tokenizer.json").write_text("{}")
    directory.joinpath("tokenizer_config.json").write_text("{}")

    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    keys = sorted(state)
    midpoint = len(keys) // 2
    save_file({key: state[key] for key in keys[:midpoint]}, directory / "model-00001.safetensors")
    save_file({key: state[key] for key in keys[midpoint:]}, directory / "model-00002.safetensors")


def file_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def checkpoint_tensors(directory: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for path in sorted(directory.glob("model*.safetensors")):
        tensors.update(load_file(path))
    return tensors


def remove_tensor(directory: Path, key: str) -> None:
    for shard_path in sorted(directory.glob("model*.safetensors")):
        tensors = load_file(shard_path)
        if key in tensors:
            tensors.pop(key)
            save_file(tensors, shard_path)
            return
    raise KeyError(key)


def add_tensor(directory: Path, key: str, tensor: torch.Tensor) -> None:
    shard_path = min(directory.glob("model*.safetensors"))
    tensors = load_file(shard_path)
    tensors[key] = tensor
    save_file(tensors, shard_path)


def replace_tensor(directory: Path, key: str, tensor: torch.Tensor) -> None:
    remove_tensor(directory, key)
    add_tensor(directory, key, tensor)


def duplicate_tensor(directory: Path, key: str, tensor: torch.Tensor) -> None:
    for shard_path in sorted(directory.glob("model*.safetensors")):
        tensors = load_file(shard_path)
        if key not in tensors:
            tensors[key] = tensor
            save_file(tensors, shard_path)
            return
    save_file({key: tensor}, directory / "model-duplicate.safetensors")


def test_q8_conversion_is_deterministic_and_preserves_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(31)
    source = tmp_path / "source"
    output_a = tmp_path / "q8-a"
    output_b = tmp_path / "q8-b"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()).eval(), source)
    source_before = file_bytes(source)
    monkeypatch.setattr(convert_module, "MAX_SHARD_BYTES", 10_000)

    convert_checkpoint(
        source,
        output_a,
        source_model="Tiny/Qwen",
        source_revision="revision-1",
    )
    convert_checkpoint(
        source,
        output_b,
        source_model="Tiny/Qwen",
        source_revision="revision-1",
    )

    metadata = QuantizationConfig.from_directory(output_a)
    tensors = checkpoint_tensors(output_a)
    assert file_bytes(source) == source_before
    assert file_bytes(output_a) == file_bytes(output_b)
    assert len(list(output_a.glob("model*.safetensors"))) > 1
    assert all(
        sum(tensor.numel() * tensor.element_size() for tensor in load_file(shard).values())
        <= 10_000
        for shard in output_a.glob("model*.safetensors")
    )
    assert metadata.quantization == "q8"
    assert metadata.group_size == 32
    assert metadata.source_model == "Tiny/Qwen"
    assert metadata.source_revision == "revision-1"
    assert metadata.tensors.count("model.embed_tokens.weight") == 1
    assert "lm_head.weight" not in tensors
    assert all(tensors[name].dtype == torch.int8 for name in metadata.tensors)
    assert all(
        tensors[name.removesuffix(".weight") + ".scales"].dtype == torch.float16
        for name in metadata.tensors
    )
    assert tensors["model.norm.weight"].dtype == torch.float32
    assert tensors["model.layers.0.self_attn.q_proj.bias"].dtype == torch.float32


def test_q8_checkpoint_loads_without_the_source_and_matches_in_memory_q8(tmp_path: Path) -> None:
    torch.manual_seed(37)
    source_model = QwenForCausalLM(tiny_q8_config()).eval()
    expected = deepcopy(source_model).quantize_q8_()
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(source_model, source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    shutil.rmtree(source)

    actual = QwenForCausalLM.from_pretrained(
        output,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 5, 7, 9]])

    assert actual.quantization_name == "q8"
    assert actual.activation_dtype == torch.float32
    assert isinstance(actual.model.embed_tokens, Q8Embedding)
    assert sum(isinstance(module, Q8Linear) for module in actual.modules()) == 14
    assert all(
        module.weight.dtype == torch.int8
        for module in actual.modules()
        if isinstance(module, Q8Linear)
    )
    assert all(
        module.scales.dtype == torch.float16
        for module in actual.modules()
        if isinstance(module, (Q8Linear, Q8Embedding))
    )
    with torch.inference_mode():
        torch.testing.assert_close(actual(input_ids), expected(input_ids))


@pytest.mark.parametrize("missing_kind", ["weight", "scale"])
def test_q8_loader_rejects_a_missing_packed_tensor(tmp_path: Path, missing_kind: str) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    weight_name = QuantizationConfig.from_directory(output).tensors[0]
    missing_key = (
        weight_name if missing_kind == "weight" else weight_name.removesuffix(".weight") + ".scales"
    )
    remove_tensor(output, missing_key)

    with pytest.raises(ValueError, match=missing_key):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_q8_loader_preserves_packed_dtypes_and_casts_residuals(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")

    model = QwenForCausalLM.from_pretrained(
        output,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    projection = model.model.layers[0].self_attn.q_proj

    assert projection.weight.dtype == torch.int8
    assert projection.scales.dtype == torch.float16
    assert projection.bias.dtype == torch.bfloat16
    assert model.model.norm.weight.dtype == torch.bfloat16
    assert model.model.embed_tokens(torch.tensor([1])).dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("kind", "dtype"),
    [("weight", torch.float32), ("scale", torch.bfloat16)],
)
def test_q8_loader_rejects_wrong_packed_dtypes(
    tmp_path: Path, kind: str, dtype: torch.dtype
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    weight_name = QuantizationConfig.from_directory(output).tensors[0]
    name = weight_name if kind == "weight" else weight_name.removesuffix(".weight") + ".scales"
    tensor = checkpoint_tensors(output)[name].to(dtype)
    replace_tensor(output, name, tensor)

    with pytest.raises(ValueError, match=rf"{name} must be torch\."):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_q8_cached_decode_matches_full_prefix_logits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    model = QwenForCausalLM.from_pretrained(
        output,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 5, 7, 9]])
    cache = create_kv_cache(
        "contiguous",
        num_layers=model.config.num_hidden_layers,
        batch_size=1,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        capacity=input_ids.shape[1],
        block_size=16,
        device=torch.device("cpu"),
        dtype=model.activation_dtype,
    )

    with torch.inference_mode():
        full_logits = model(input_ids)
        model(input_ids[:, :3], cache=cache)
        cached_logits = model(input_ids[:, 3:], cache=cache, position=3)

    torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], rtol=1e-5, atol=1e-5)


def test_q8_loader_rejects_metadata_that_does_not_match_the_model(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    metadata_path = output / "quantization_config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tensors"].pop()
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="metadata does not match"):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_q8_loader_rejects_a_duplicate_floating_output_head(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    shard_path = min(output.glob("model*.safetensors"))
    tensors = load_file(shard_path)
    tensors["lm_head.weight"] = torch.zeros((32, 32))
    save_file(tensors, shard_path)

    with pytest.raises(ValueError, match="lm_head.weight"):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_q8_loader_rejects_a_tensor_repeated_across_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    shard_paths = sorted(output.glob("model*.safetensors"))
    first_shard = load_file(shard_paths[0])
    name, tensor = next(iter(first_shard.items()))
    duplicate_tensor(output, name, tensor)

    with pytest.raises(ValueError, match=rf"repeats tensors.*{name}"):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_q8_loader_rejects_mps_without_custom_shader_support(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    monkeypatch.delattr(torch.mps, "compile_shader", raising=False)

    with pytest.raises(RuntimeError, match="torch.mps.compile_shader"):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("mps"),
            dtype=torch.bfloat16,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_q8_checkpoint_runs_the_fused_model_on_mps(tmp_path: Path) -> None:
    torch.manual_seed(29)
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")
    cpu_model = QwenForCausalLM.from_pretrained(
        output,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    mps_model = QwenForCausalLM.from_pretrained(
        output,
        device=torch.device("mps"),
        dtype=torch.bfloat16,
    )
    input_ids = torch.tensor([[1, 5, 7, 9]])

    with torch.inference_mode():
        expected = cpu_model(input_ids).to(torch.bfloat16)
        actual = mps_model(input_ids.to("mps")).cpu()

    assert all(
        module.weight.device.type == "mps"
        for module in mps_model.modules()
        if isinstance(module, Q8Linear)
    )
    # Operator parity is tighter; this allows BF16 rounding through the complete tiny model.
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_q8_checkpoint_requires_bfloat16_activations_on_mps(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    convert_checkpoint(source, output, source_model="Tiny/Qwen")

    with pytest.raises(ValueError, match="bfloat16"):
        QwenForCausalLM.from_pretrained(
            output,
            device=torch.device("mps"),
            dtype=torch.float16,
        )


def test_failed_conversion_leaves_no_output_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    missing_key = "model.layers.0.self_attn.q_proj.weight"
    remove_tensor(source, missing_key)

    with pytest.raises(ValueError, match=missing_key):
        convert_checkpoint(source, output, source_model="Tiny/Qwen")

    assert not output.exists()
    assert not list(tmp_path.glob(".q8.*"))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("missing", "model.norm.weight"),
        ("unexpected", "unexpected.weight"),
        ("duplicate", "model.norm.weight"),
    ],
)
def test_converter_rejects_an_invalid_source_state(
    tmp_path: Path, change: str, expected: str
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    tensor = checkpoint_tensors(source)["model.norm.weight"]
    if change == "missing":
        remove_tensor(source, expected)
    elif change == "unexpected":
        add_tensor(source, expected, tensor)
    else:
        duplicate_tensor(source, expected, tensor)

    with pytest.raises(ValueError, match=expected):
        convert_checkpoint(source, output, source_model="Tiny/Qwen")

    assert not output.exists()


def test_converter_refuses_to_overwrite_an_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "q8"
    write_checkpoint(QwenForCausalLM(tiny_q8_config()), source)
    output.mkdir()

    with pytest.raises(FileExistsError, match=str(output)):
        convert_checkpoint(source, output, source_model="Tiny/Qwen")
