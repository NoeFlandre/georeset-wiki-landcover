"""Loader utilities for article land-use evidence metadata."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from georeset.utils.json_io import read_json_file

_EVIDENCE_METADATA_COLUMNS = [
    "pageid",
    "landcover_relevance",
    "uncertainty",
    "evidence_types",
    "evidence_sentences_count",
    "landuse_evidence_summary_char_count",
]


def _normalize_list_value(value: object) -> list[str]:
    """Normalize a single value into a list of strings."""
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

    if isinstance(value, list):
        return normalize_items(value)
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
        if isinstance(parsed, tuple):
            return normalize_items(parsed)
        if isinstance(parsed, list):
            return normalize_items(parsed)
        if parsed is None:
            return []
        return normalize_items([parsed])
    return normalize_items([value])


def _coerce_count(value: object) -> int:
    """Coerce a numeric-like value to int, defaulting to 0."""
    raw = pd.to_numeric(value, errors="coerce")
    if pd.isna(raw):
        return 0
    return int(raw)


def load_evidence_metadata(path: Path) -> pd.DataFrame:
    """Load evidence metadata records from a JSON mapping."""
    raw = read_json_file(path)
    if not isinstance(raw, dict):
        return pd.DataFrame(columns=_EVIDENCE_METADATA_COLUMNS)

    rows: list[dict[str, object]] = []
    for pageid_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        pageid = payload.get("pageid", pageid_key)
        if pageid in (None, ""):
            pageid = pageid_key
        row = {
            "pageid": str(pageid),
            "landcover_relevance": payload.get("landcover_relevance"),
            "uncertainty": payload.get("uncertainty"),
            "evidence_types": _normalize_list_value(payload.get("evidence_types")),
            "evidence_sentences_count": _coerce_count(payload.get("evidence_sentences_count")),
            "landuse_evidence_summary_char_count": _coerce_count(
                payload.get("landuse_evidence_summary_char_count")
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_EVIDENCE_METADATA_COLUMNS)
    return pd.DataFrame(rows)
