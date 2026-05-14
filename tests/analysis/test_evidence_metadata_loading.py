"""Tests for shared evidence metadata loading."""

from __future__ import annotations

import json
import math
from pathlib import Path

from georeset.analysis.evidence_metadata_loading import load_evidence_metadata


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_evidence_metadata_normalizes_pageid_from_payload_or_key(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(
        payload_path,
        {
            "1": {"pageid": 10, "landcover_relevance": "low"},
            "2": {"landcover_relevance": "medium", "uncertainty": "high"},
            "3": "ignore me",
        },
    )

    frame = load_evidence_metadata(payload_path)

    assert list(frame["pageid"]) == ["10", "2"]


def test_load_evidence_metadata_falls_back_to_key_for_none_or_empty_pageid(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(
        payload_path,
        {
            "abc": {"pageid": None},
            "def": {"pageid": ""},
        },
    )

    frame = load_evidence_metadata(payload_path)

    assert list(frame["pageid"]) == ["abc", "def"]


def test_load_evidence_metadata_ignores_non_dict_payloads(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(
        payload_path,
        {
            "1": ["not", "a", "dict"],
            "2": None,
            "3": {"landcover_relevance": "low"},
        },
    )

    frame = load_evidence_metadata(payload_path)

    assert frame["pageid"].tolist() == ["3"]


def test_load_evidence_metadata_returns_stable_columns_for_no_usable_records(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(payload_path, {"1": "not-a-record", "2": 4, "3": [1, 2, 3]})
    frame = load_evidence_metadata(payload_path)

    assert frame.empty
    assert list(frame.columns) == [
        "pageid",
        "landcover_relevance",
        "uncertainty",
        "evidence_types",
        "evidence_sentences_count",
        "landuse_evidence_summary_char_count",
    ]


def test_load_evidence_metadata_returns_stable_columns_for_non_dict_top_level_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(payload_path, [{"pageid": "1", "evidence_types": ["forest"]}])

    frame = load_evidence_metadata(payload_path)

    assert frame.empty
    assert list(frame.columns) == [
        "pageid",
        "landcover_relevance",
        "uncertainty",
        "evidence_types",
        "evidence_sentences_count",
        "landuse_evidence_summary_char_count",
    ]


def test_load_evidence_metadata_normalizes_evidence_types_to_list_of_non_empty_strings(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(
        payload_path,
        {
            "1": {
                "pageid": "1",
                "evidence_types": ["forest", 2, None, "", "   ", math.nan],
            }
        },
    )

    frame = load_evidence_metadata(payload_path)

    assert frame.loc[0, "evidence_types"] == ["forest", "2"]


def test_load_evidence_metadata_parses_stringified_evidence_types(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(
        payload_path,
        {
            "1": {"pageid": "1", "evidence_types": '["forest", "water", null, ""]'},
            "2": {"pageid": "2", "evidence_types": "['urban', 'wetland']"},
            "3": {"pageid": "3", "evidence_types": "forest"},
            "4": {"pageid": "4", "evidence_types": ""},
        },
    )

    frame = load_evidence_metadata(payload_path)

    assert frame["evidence_types"].tolist() == [
        ["forest", "water"],
        ["urban", "wetland"],
        ["forest"],
        [],
    ]


def test_load_evidence_metadata_numeric_fields_default_to_zero_on_bad_or_missing(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(
        payload_path,
        {
            "1": {
                "pageid": "1",
                "evidence_sentences_count": "3",
                "landuse_evidence_summary_char_count": 7.2,
            },
            "2": {"pageid": "2", "evidence_sentences_count": "bad", "landuse_evidence_summary_char_count": None},
            "3": {"pageid": "3", "evidence_sentences_count": None, "landuse_evidence_summary_char_count": "nope"},
            "4": {"pageid": "4"},
        },
    )

    frame = load_evidence_metadata(payload_path)

    assert frame.loc[0, "evidence_sentences_count"] == 3
    assert frame.loc[0, "landuse_evidence_summary_char_count"] == 7
    assert frame.loc[1, "evidence_sentences_count"] == 0
    assert frame.loc[1, "landuse_evidence_summary_char_count"] == 0
    assert frame.loc[2, "evidence_sentences_count"] == 0
    assert frame.loc[2, "landuse_evidence_summary_char_count"] == 0
    assert frame.loc[3, "evidence_sentences_count"] == 0
    assert frame.loc[3, "landuse_evidence_summary_char_count"] == 0


def test_load_evidence_metadata_preserves_raw_relevance_values(tmp_path: Path) -> None:
    payload_path = tmp_path / "evidence_metadata.json"
    _write_json(payload_path, {"1": {"pageid": "1", "landcover_relevance": "LOW", "uncertainty": "HIGH"}})

    frame = load_evidence_metadata(payload_path)

    assert frame.loc[0, "landcover_relevance"] == "LOW"
    assert frame.loc[0, "uncertainty"] == "HIGH"
