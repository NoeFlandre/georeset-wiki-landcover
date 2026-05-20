"""Shared access helpers for text artifact records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def mapping_get(
    mapping: Mapping[str, Any] | pd.Series | None, key: str, default: Any = None
) -> Any:
    """Read a key from a JSON-like mapping or pandas row."""
    if mapping is None:
        return default
    return mapping.get(key, default)


def is_missing(value: object) -> bool:
    """Return whether a record value should be treated as absent."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return False
    return bool(pd.isna(value))


def json_scalar(value: Any, *, default: Any = None) -> Any:
    """Normalize pandas/numpy scalar values for JSON records."""
    if is_missing(value):
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):
        converted = value.item()
        return json_scalar(converted, default=default)
    return value
