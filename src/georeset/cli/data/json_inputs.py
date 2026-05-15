"""Shared JSON input helpers for data-building CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from georeset.utils.json_io import read_json_file


def read_optional_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = read_json_file(path)
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, Any], raw)


def read_required_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    raw = read_json_file(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{description} file '{path}' must contain a JSON object.")
    return cast(dict[str, Any], raw)


def index_json_records_by_pageid(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for pageid_key, payload in records.items():
        if not isinstance(payload, dict):
            continue
        pageid = payload.get("pageid", pageid_key)
        if pageid in (None, ""):
            pageid = pageid_key
        indexed[str(pageid)] = payload
    return indexed
