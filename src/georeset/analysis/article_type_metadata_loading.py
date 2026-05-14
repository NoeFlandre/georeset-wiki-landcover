"""Loader utilities for article-type metadata."""

from __future__ import annotations

import ast
import json
from numbers import Real
from pathlib import Path

import pandas as pd

from georeset.utils.json_io import read_json_file

_ARTICLE_TYPE_METADATA_COLUMNS = [
    "pageid",
    "title",
    "primary_article_type",
    "candidate_article_types",
    "matched_categories",
    "matched_rules",
    "all_categories_count",
    "has_categories",
]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return False
    return bool(pd.isna(value))


def _coalesce_text(value: object, default: str = "") -> str:
    if _is_missing(value):
        return default
    return str(value)


def _coalesce_pageid(value: object, fallback: object) -> str | None:
    if not _is_missing(value):
        return str(value)
    if _is_missing(fallback):
        return None
    return str(fallback)


def _normalize_list_value(value: object) -> list[str]:
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
    if pd.isna(value):
        return []
    return normalize_items([value])


def _coerce_count(value: object) -> int:
    raw = pd.to_numeric(value, errors="coerce")
    if pd.isna(raw):
        return 0
    return int(raw)


def _coerce_bool(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, dict)):
        return False
    if pd.isna(value):
        return False
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in {"true", "1", "yes", "y", "on"}:
            return True
        if value_lower in {"false", "0", "no", "n", "off"}:
            return False
        return False
    if isinstance(value, Real):
        if value == 1:
            return True
        if value == 0:
            return False
        return False
    return False


def load_article_type_metadata(path: Path) -> pd.DataFrame:
    """Load article-type metadata from JSON mapping or CSV row data."""
    if not path.exists() or not path.is_file():
        return pd.DataFrame(columns=_ARTICLE_TYPE_METADATA_COLUMNS)

    if path.suffix.lower() == ".json":
        raw = read_json_file(path)
        if not isinstance(raw, dict):
            return pd.DataFrame(columns=_ARTICLE_TYPE_METADATA_COLUMNS)
        json_rows: list[dict[str, object]] = []
        for pageid_key, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            pageid = _coalesce_pageid(payload.get("pageid"), pageid_key)
            if pageid is None:
                continue
            candidate_article_types = _normalize_list_value(
                payload.get("candidate_article_types", ["other_or_unclear"])
            )
            if not candidate_article_types:
                candidate_article_types = ["other_or_unclear"]
            json_rows.append(
                {
                    "pageid": pageid,
                    "title": _coalesce_text(payload.get("title", "")),
                    "primary_article_type": _coalesce_text(
                        payload.get("primary_article_type", "other_or_unclear"),
                        default="other_or_unclear",
                    ),
                    "candidate_article_types": candidate_article_types,
                    "matched_categories": _normalize_list_value(
                        payload.get("matched_categories", [])
                    ),
                    "matched_rules": _normalize_list_value(payload.get("matched_rules", [])),
                    "all_categories_count": _coerce_count(payload.get("all_categories_count", 0)),
                    "has_categories": _coerce_bool(payload.get("has_categories", False)),
                }
            )
        if not json_rows:
            return pd.DataFrame(columns=_ARTICLE_TYPE_METADATA_COLUMNS)
        return pd.DataFrame(json_rows)

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str)
        if frame.empty:
            return pd.DataFrame(columns=_ARTICLE_TYPE_METADATA_COLUMNS)
        csv_rows: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            pageid = _coalesce_pageid(row.get("pageid"), None)
            if pageid is None:
                continue
            candidate_article_types = _normalize_list_value(row.get("candidate_article_types"))
            if not candidate_article_types:
                candidate_article_types = ["other_or_unclear"]
            csv_rows.append(
                {
                    "pageid": pageid,
                    "title": _coalesce_text(row.get("title", "")),
                    "primary_article_type": _coalesce_text(
                        row.get("primary_article_type", "other_or_unclear"),
                        default="other_or_unclear",
                    ),
                    "candidate_article_types": candidate_article_types,
                    "matched_categories": _normalize_list_value(row.get("matched_categories", [])),
                    "matched_rules": _normalize_list_value(row.get("matched_rules", [])),
                    "all_categories_count": _coerce_count(row.get("all_categories_count", 0)),
                    "has_categories": _coerce_bool(row.get("has_categories", False)),
                }
            )
        if not csv_rows:
            return pd.DataFrame(columns=_ARTICLE_TYPE_METADATA_COLUMNS)
        return pd.DataFrame(csv_rows)

    return pd.DataFrame(columns=_ARTICLE_TYPE_METADATA_COLUMNS)
