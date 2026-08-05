"""Explicit runtime choices shared by the CLI and engine."""

from tinyinfer.attention import ATTENTION_NAMES, DEFAULT_ATTENTION
from tinyinfer.kv_cache import KV_CACHE_NAMES

DEFAULT_KV_CACHE = "contiguous"

__all__ = ["ATTENTION_NAMES", "DEFAULT_ATTENTION", "DEFAULT_KV_CACHE", "KV_CACHE_NAMES"]
