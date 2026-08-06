import importlib

import pytest

ROADMAP_MODULES = (
    "tinyinfer.runtime",
    "tinyinfer.kv_cache.none",
    "tinyinfer.kv_cache.contiguous",
    "tinyinfer.kv_cache.paged",
    "tinyinfer.attention.eager",
    "tinyinfer.attention.sdpa",
    "tinyinfer.attention.flash",
    "tinyinfer.decoding.autoregressive",
    "tinyinfer.decoding.speculative",
    "tinyinfer.decoding.dspark.drafter",
    "tinyinfer.decoding.dspark.confidence",
    "tinyinfer.decoding.dspark.scheduler",
    "tinyinfer.scheduling.serial",
    "tinyinfer.scheduling.continuous",
    "tinyinfer.sampling.greedy",
    "tinyinfer.sampling.top_k",
    "tinyinfer.sampling.top_p",
    "tinyinfer.quantization.int8",
    "tinyinfer.quantization.int4",
    "tinyinfer.quantization.linear",
    "tinyinfer.parallelism.tensor",
    "tinyinfer.parallelism.pipeline",
)


@pytest.mark.parametrize("module_name", ROADMAP_MODULES)
def test_optimization_roadmap_module_is_importable(module_name: str) -> None:
    importlib.import_module(module_name)
