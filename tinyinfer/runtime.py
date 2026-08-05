"""Explicit runtime choices shared by the CLI and engine."""

from tinyinfer.attention import ATTENTION_NAMES, DEFAULT_ATTENTION
from tinyinfer.decoding import DECODING_NAMES
from tinyinfer.kv_cache import KV_CACHE_NAMES

DEFAULT_DECODING = "autoregressive"
DEFAULT_KV_CACHE = "contiguous"

__all__ = [
    "ATTENTION_NAMES",
    "DECODING_NAMES",
    "DEFAULT_ATTENTION",
    "DEFAULT_DECODING",
    "DEFAULT_KV_CACHE",
    "KV_CACHE_NAMES",
]
