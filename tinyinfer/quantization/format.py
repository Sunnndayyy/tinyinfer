"""Versioned metadata for TinyInfer weight-only checkpoints."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

FORMAT_FILE = "quantization_config.json"
FORMAT_NAME = "tinyinfer-weight-only"
FORMAT_VERSION = 1


def q8_weight_names(num_layers: int) -> tuple[str, ...]:
    names = ["model.embed_tokens.weight"]
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        names.extend(
            f"{prefix}.{name}.weight"
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
    return tuple(sorted(names))


def scale_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight") + ".scales"


@dataclass(frozen=True)
class QuantizationConfig:
    format: str
    version: int
    quantization: str
    bits: int
    group_size: int
    scale_dtype: str
    symmetric: bool
    packing_order: str
    source_model: str
    source_revision: str | None
    tensors: tuple[str, ...]

    @classmethod
    def q8(
        cls,
        *,
        source_model: str,
        source_revision: str | None,
        tensors: tuple[str, ...],
        group_size: int = 32,
    ) -> "QuantizationConfig":
        return cls(
            format=FORMAT_NAME,
            version=FORMAT_VERSION,
            quantization="q8",
            bits=8,
            group_size=group_size,
            scale_dtype="float16",
            symmetric=True,
            packing_order="signed-int8",
            source_model=source_model,
            source_revision=source_revision,
            tensors=tensors,
        )

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "QuantizationConfig":
        path = Path(model_dir) / FORMAT_FILE
        values = json.loads(path.read_text())
        values["tensors"] = tuple(values["tensors"])
        config = cls(**values)
        if (
            config.format != FORMAT_NAME
            or config.version != FORMAT_VERSION
            or config.quantization != "q8"
            or config.bits != 8
            or config.group_size != 32
            or config.scale_dtype != "float16"
            or not config.symmetric
            or config.packing_order != "signed-int8"
        ):
            raise ValueError(f"unsupported quantization metadata in {path}")
        return config

    def write(self, model_dir: str | Path) -> None:
        values = asdict(self)
        values["tensors"] = list(self.tensors)
        Path(model_dir, FORMAT_FILE).write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def read_quantization_config(model_dir: str | Path) -> QuantizationConfig | None:
    path = Path(model_dir) / FORMAT_FILE
    return QuantizationConfig.from_directory(model_dir) if path.is_file() else None
