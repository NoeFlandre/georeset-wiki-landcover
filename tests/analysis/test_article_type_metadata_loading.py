"""Tests for shared article-type metadata loading."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from georeset.analysis.article_type_metadata_loading import load_article_type_metadata


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_article_type_metadata_missing_path_returns_stable_empty_df(tmp_path: Path) -> None:
    frame = load_article_type_metadata(tmp_path / "missing_article_type_metadata.json")

    assert frame.empty
    assert list(frame.columns) == [
        "pageid",
        "title",
        "primary_article_type",
        "candidate_article_types",
        "matched_categories",
        "matched_rules",
        "all_categories_count",
        "has_categories",
    ]


def test_load_article_type_metadata_unsupported_suffix_returns_stable_empty(tmp_path: Path) -> None:
    path = tmp_path / "article_types.txt"
    path.write_text("noop", encoding="utf-8")

    frame = load_article_type_metadata(path)

    assert frame.empty
    assert list(frame.columns) == [
        "pageid",
        "title",
        "primary_article_type",
        "candidate_article_types",
        "matched_categories",
        "matched_rules",
        "all_categories_count",
        "has_categories",
    ]


def test_load_article_type_metadata_non_dict_top_level_returns_stable_empty(tmp_path: Path) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(path, ["not", "a", "dict"])

    frame = load_article_type_metadata(path)

    assert frame.empty
    assert list(frame.columns) == [
        "pageid",
        "title",
        "primary_article_type",
        "candidate_article_types",
        "matched_categories",
        "matched_rules",
        "all_categories_count",
        "has_categories",
    ]


def test_load_article_type_metadata_non_dict_records_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "1": {"pageid": 1, "primary_article_type": "water_feature"},
            "2": "ignore",
            "3": None,
            "4": [1, 2, 3],
        },
    )

    frame = load_article_type_metadata(path)

    assert frame["pageid"].tolist() == ["1"]


def test_load_article_type_metadata_ignores_non_dict_records_without_rows(tmp_path: Path) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "1": "ignore me",
            "2": [1, 2, 3],
            "3": None,
        },
    )

    frame = load_article_type_metadata(path)

    assert frame.empty
    assert list(frame.columns) == [
        "pageid",
        "title",
        "primary_article_type",
        "candidate_article_types",
        "matched_categories",
        "matched_rules",
        "all_categories_count",
        "has_categories",
    ]


def test_load_article_type_metadata_normalizes_pageid_from_payload_or_key(tmp_path: Path) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "100": {"pageid": 10, "primary_article_type": "water_feature"},
            "200": {"pageid": None, "primary_article_type": "water_feature"},
            "300": {"pageid": "", "primary_article_type": "water_feature"},
            "400": {"primary_article_type": "water_feature"},
            "500": {"pageid": float("nan"), "primary_article_type": "water_feature"},
        },
    )

    frame = load_article_type_metadata(path)

    assert frame["pageid"].tolist() == ["10", "200", "300", "400", "500"]


def test_load_article_type_metadata_normalizes_candidate_and_rule_categories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "1": {
                "candidate_article_types": '["water_feature", "wetland", null, ""]',
                "matched_categories": "['forêt', 'rivière']",
                "matched_rules": "('rule-a', 'rule-b')",
                "primary_article_type": "water_feature",
            },
            "2": {
                "candidate_article_types": "water_feature",
                "matched_categories": "",
                "matched_rules": None,
                "primary_article_type": "water_feature",
            },
            "3": {
                "candidate_article_types": [None, "", "   ", math.nan],
                "matched_categories": ["forêt", None, "", math.nan],
                "matched_rules": ["rule-a", None, ""],
                "primary_article_type": "water_feature",
            },
        },
    )

    frame = load_article_type_metadata(path)

    assert frame.loc[0, "candidate_article_types"] == ["water_feature", "wetland"]
    assert frame.loc[0, "matched_categories"] == ["forêt", "rivière"]
    assert frame.loc[0, "matched_rules"] == ["rule-a", "rule-b"]
    assert frame.loc[1, "candidate_article_types"] == ["water_feature"]
    assert frame.loc[1, "matched_categories"] == []
    assert frame.loc[1, "matched_rules"] == []
    assert frame.loc[2, "candidate_article_types"] == ["other_or_unclear"]


def test_load_article_type_metadata_all_categories_count_and_has_categories_are_normalized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "1": {
                "all_categories_count": "3",
                "has_categories": "true",
                "primary_article_type": "water_feature",
            },
            "2": {
                "all_categories_count": 2.7,
                "has_categories": "0",
                "primary_article_type": "water_feature",
            },
            "3": {
                "all_categories_count": "bad",
                "has_categories": "False",
                "primary_article_type": "water_feature",
            },
            "4": {"has_categories": None, "primary_article_type": "water_feature"},
            "5": {"has_categories": "oui", "primary_article_type": "water_feature"},
            "6": {"has_categories": "non", "primary_article_type": "water_feature"},
        },
    )

    frame = load_article_type_metadata(path)

    assert frame.loc[0, "all_categories_count"] == 3
    assert frame.loc[1, "all_categories_count"] == 2
    assert frame.loc[2, "all_categories_count"] == 0
    assert frame.loc[3, "all_categories_count"] == 0
    assert frame["has_categories"].tolist() == [True, False, False, False, True, False]


def test_load_article_type_metadata_unknown_has_categories_string_is_false(tmp_path: Path) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "1": {"has_categories": "maybe", "primary_article_type": "water_feature"},
            "2": {"has_categories": "unknown", "primary_article_type": "water_feature"},
        },
    )

    frame = load_article_type_metadata(path)

    assert frame["has_categories"].tolist() == [False, False]


def test_load_article_type_metadata_container_has_categories_values_are_false(
    tmp_path: Path,
) -> None:
    path = tmp_path / "article_type_metadata.json"
    _write_json(
        path,
        {
            "1": {"has_categories": ["a"], "primary_article_type": "water_feature"},
            "2": {"has_categories": {"a": "b"}, "primary_article_type": "water_feature"},
        },
    )

    frame = load_article_type_metadata(path)

    assert frame["has_categories"].tolist() == [False, False]


def test_load_article_type_metadata_supports_csv_input(tmp_path: Path) -> None:
    path = tmp_path / "article_type_assignments.csv"
    _write_csv(
        path,
        [
            {
                "pageid": "10",
                "title": "Alpha",
                "primary_article_type": "water_feature",
                "candidate_article_types": "['water_feature', 'wetland']",
                "matched_categories": "['a', 'b']",
                "matched_rules": "('rule-a', 'rule-b')",
                "all_categories_count": "4",
                "has_categories": "1",
            },
            {
                "pageid": "",
                "title": "Missing",
                "primary_article_type": "water_feature",
                "candidate_article_types": "['water_feature']",
            },
            {
                "pageid": "11",
                "title": "Beta",
                "primary_article_type": "other_or_unclear",
                "candidate_article_types": "",
                "matched_categories": "",
                "matched_rules": "",
                "all_categories_count": "bad",
                "has_categories": "false",
            },
        ],
    )

    frame = load_article_type_metadata(path)

    assert len(frame) == 2
    assert frame.iloc[0]["pageid"] == "10"
    assert frame.iloc[0]["candidate_article_types"] == ["water_feature", "wetland"]
    assert frame.iloc[0]["all_categories_count"] == 4
    assert bool(frame.iloc[0]["has_categories"]) is True
    assert frame.iloc[1]["pageid"] == "11"
    assert frame.iloc[1]["candidate_article_types"] == ["other_or_unclear"]
    assert frame.iloc[1]["matched_categories"] == []
    assert bool(frame.iloc[1]["has_categories"]) is False
