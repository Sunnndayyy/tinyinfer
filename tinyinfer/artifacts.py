from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
REQUIRED_MODEL_FILES = (*MODEL_METADATA_FILES, "model*.safetensors")


def resolve_model(model: str, cache_dir: str | Path | None = None) -> Path:
    """Return a local model directory, downloading a Hub model when needed."""
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return local_path.resolve()

    downloaded = snapshot_download(
        repo_id=model,
        cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
        allow_patterns=list(REQUIRED_MODEL_FILES),
    )
    return Path(downloaded)


def weight_shards(model_dir: str | Path) -> list[Path]:
    shards = sorted(Path(model_dir).glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model safetensors found in {model_dir}")
    return shards
