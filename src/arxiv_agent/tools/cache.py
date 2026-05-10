"""Simple file-based cache for arXiv API responses and PDFs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from arxiv_agent.config import CACHE_DIR


def _cache_path(namespace: str, key: str, suffix: str = "json") -> Path:
    """Return the cache file path for a given namespace + key."""
    safe_key = hashlib.sha256(key.encode()).hexdigest()[:32]
    namespace_dir = CACHE_DIR / namespace
    namespace_dir.mkdir(exist_ok=True)
    return namespace_dir / f"{safe_key}.{suffix}"


def cache_get_json(namespace: str, key: str) -> Any | None:
    """Return cached JSON for (namespace, key) or None if not cached."""
    path = _cache_path(namespace, key, suffix="json")
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def cache_set_json(namespace: str, key: str, value: Any) -> None:
    """Write a JSON-serializable value to the cache."""
    path = _cache_path(namespace, key, suffix="json")
    with path.open("w") as f:
        json.dump(value, f, default=str)


def cache_get_bytes(namespace: str, key: str) -> bytes | None:
    """Return cached bytes for (namespace, key) or None."""
    path = _cache_path(namespace, key, suffix="bin")
    if not path.exists():
        return None
    return path.read_bytes()


def cache_set_bytes(namespace: str, key: str, value: bytes) -> None:
    """Write bytes to the cache."""
    path = _cache_path(namespace, key, suffix="bin")
    path.write_bytes(value)