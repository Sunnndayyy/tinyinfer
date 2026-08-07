"""Explicit runtime choices shared by the CLI and engine."""

from tinyinfer.attention import ATTENTION_NAMES, DEFAULT_ATTENTION
from tinyinfer.decoding import DECODING_NAMES
from tinyinfer.kv_cache import KV_CACHE_NAMES

DEFAULT_DECODING = "autoregressive"
DEFAULT_KV_CACHE = "contiguous"
DEFAULT_QUANTIZATION = "auto"
QUANTIZATION_NAMES = ("none", "q8", "q4")

__all__ = [
    "ATTENTION_NAMES",
    "DECODING_NAMES",
    "DEFAULT_ATTENTION",
    "DEFAULT_DECODING",
    "DEFAULT_KV_CACHE",
    "DEFAULT_QUANTIZATION",
    "KV_CACHE_NAMES",
    "QUANTIZATION_NAMES",
]
