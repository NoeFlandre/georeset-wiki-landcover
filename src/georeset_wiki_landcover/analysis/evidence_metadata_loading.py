"""Loader utilities for article land-use evidence metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from georeset_wiki_landcover.analysis.list_normalization import normalize_string_list
from georeset_wiki_landcover.utils.json_io import read_json_file

_EVIDENCE_METADATA_COLUMNS = [
    "pageid",
    "landcover_relevance",
    "uncertainty",
    "evidence_types",
    "evidence_sentences_count",
    "landuse_evidence_summary_char_count",
]


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
            "evidence_types": normalize_string_list(payload.get("evidence_types")),
            "evidence_sentences_count": _coerce_count(payload.get("evidence_sentences_count")),
            "landuse_evidence_summary_char_count": _coerce_count(
                payload.get("landuse_evidence_summary_char_count")
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_EVIDENCE_METADATA_COLUMNS)
    return pd.DataFrame(rows)
