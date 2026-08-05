"""Explicit runtime choices shared by the CLI and engine."""

from tinyinfer.kv_cache import KV_CACHE_NAMES

DEFAULT_KV_CACHE = "contiguous"

__all__ = ["DEFAULT_KV_CACHE", "KV_CACHE_NAMES"]
