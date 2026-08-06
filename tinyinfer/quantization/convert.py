"""Offline Q8 checkpoint conversion."""

import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tinyinfer.artifacts import MODEL_METADATA_FILES, weight_shards
from tinyinfer.model import QwenConfig, QwenForCausalLM

from .format import QuantizationConfig, q8_weight_names, scale_name
from .int8 import GROUP_SIZE, quantize_q8

MAX_SHARD_BYTES = 1_000_000_000


def _tensor_bytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def convert_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    source_model: str,
    source_revision: str | None = None,
    format_name: str = "q8",
    group_size: int = GROUP_SIZE,
) -> Path:
    """Convert one source shard at a time, then publish the complete directory atomically."""
    source = Path(source_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if format_name != "q8" or group_size != GROUP_SIZE:
        raise ValueError("TinyInfer currently supports Q8 checkpoints with group size 32")
    for name in MODEL_METADATA_FILES:
        if not source.joinpath(name).is_file():
            raise FileNotFoundError(f"missing source artifact: {source / name}")

    config = QwenConfig.from_directory(source)
    source_shards = weight_shards(source)
    with torch.device("meta"):
        required_source_names = set(QwenForCausalLM(config).state_dict())
    accepted_source_names = required_source_names | {"lm_head.weight"}
    expected_weights = set(q8_weight_names(config.num_hidden_layers))
    converted_weights: set[str] = set()
    source_names: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))

    try:
        for name in MODEL_METADATA_FILES:
            shutil.copy2(source / name, temporary / name)

        converted = {}
        converted_bytes = 0
        shard_number = 1

        def flush_shard() -> None:
            nonlocal converted, converted_bytes, shard_number
            if not converted:
                return
            save_file(converted, temporary / f"model-{shard_number:05d}.safetensors")
            converted = {}
            converted_bytes = 0
            shard_number += 1

        for shard_path in source_shards:
            # Read one source tensor at a time so a large source shard is never loaded whole.
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for name in sorted(shard.keys()):
                    if name in source_names:
                        raise ValueError(f"source checkpoint repeats tensor: {name}")
                    source_names.add(name)
                    tensor = shard.get_tensor(name)
                    if name == "lm_head.weight":
                        continue  # The embedding is the tied output matrix, so store it once.
                    if name in expected_weights:
                        weight, scales = quantize_q8(tensor)
                        tensors = {name: weight, scale_name(name): scales}
                        converted_weights.add(name)
                    else:
                        tensors = {name: tensor}

                    item_bytes = sum(_tensor_bytes(value) for value in tensors.values())
                    if converted and converted_bytes + item_bytes > MAX_SHARD_BYTES:
                        flush_shard()
                    converted.update(tensors)
                    converted_bytes += item_bytes

        flush_shard()

        missing_source = required_source_names - source_names
        unexpected_source = source_names - accepted_source_names
        if missing_source or unexpected_source:
            raise ValueError(
                "source checkpoint does not match TinyInfer's Qwen2 implementation: "
                f"missing={sorted(missing_source)}, unexpected={sorted(unexpected_source)}"
            )
        missing = expected_weights - converted_weights
        if missing:
            raise ValueError(f"source checkpoint is missing Q8 weights: {sorted(missing)}")
        QuantizationConfig.q8(
            source_model=source_model,
            source_revision=source_revision,
            tensors=tuple(sorted(converted_weights)),
            group_size=group_size,
        ).write(temporary)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return output
