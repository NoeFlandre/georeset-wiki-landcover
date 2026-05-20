"""Small normalization helpers shared by analysis loaders."""

from __future__ import annotations

import ast
import json

import pandas as pd


def normalize_string_list(value: object) -> list[str]:
    """Normalize JSON/CSV list-like values into non-empty strings."""

    def normalize_items(items: list[object] | tuple[object, ...]) -> list[str]:
        output: list[str] = []
        for item in items:
            if item is None:
                continue
            if not isinstance(item, (list, tuple, dict)) and pd.isna(item):
                continue
            text = str(item).strip()
            if text:
                output.append(text)
        return output

    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        try:
            parsed = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return [cleaned]
        if isinstance(parsed, list):
            return normalize_items(parsed)
        if isinstance(parsed, tuple):
            return normalize_items(parsed)
        if parsed is None:
            return []
        return normalize_items([parsed])
    if isinstance(value, list):
        return normalize_items(value)
    if isinstance(value, tuple):
        return normalize_items(value)
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return []
    return normalize_items([value])
