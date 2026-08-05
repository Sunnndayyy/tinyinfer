"""Explicit runtime choices shared by the CLI and engine."""

from tinyinfer.decoding import DECODING_NAMES
from tinyinfer.kv_cache import KV_CACHE_NAMES

DEFAULT_DECODING = "autoregressive"
DEFAULT_KV_CACHE = "contiguous"

__all__ = ["DECODING_NAMES", "DEFAULT_DECODING", "DEFAULT_KV_CACHE", "KV_CACHE_NAMES"]
