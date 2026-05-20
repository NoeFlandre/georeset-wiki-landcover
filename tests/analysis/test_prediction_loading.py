"""Tests for shared frozen prediction loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from georeset_wiki_landcover.analysis.prediction_loading import (
    infer_model_from_metadata,
    load_annotated_prediction_records,
    load_prediction_records,
    prediction_identity,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_identity_parses_known_task_and_text_source_patterns() -> None:
    assert prediction_identity(Path("corine_level2_summary_predictions.json")) == (
        "corine_level2",
        "summary",
    )
    assert prediction_identity(Path("osm_content_shuffled_predictions.json")) == (
        "osm",
        "content_shuffled",
    )


def test_prediction_identity_rejects_unknown_filenames() -> None:
    with pytest.raises(ValueError, match=r"Unknown prediction file name: other.json"):
        prediction_identity(Path("other.json"))


def test_load_prediction_records_normalizes_pageid_to_string(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "pageid"
    corine_path = experiment_dir / "corine_level2_summary_predictions.json"
    osm_path = experiment_dir / "osm_content_predictions.json"
    experiment_dir.mkdir()
    _write_json(corine_path, {"10": {"pageid": 10, "target": "31"}})
    _write_json(osm_path, {"20": {"pageid": "20", "target": "wood"}})

    loaded = load_prediction_records(experiment_dir)

    assert set(loaded["pageid"]) == {"10", "20"}


def test_load_prediction_records_preserves_core_fields(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "core"
    experiment_dir.mkdir()
    path = experiment_dir / "corine_level2_summary_predictions.json"
    _write_json(
        path,
        {
            "1": {
                "pageid": 1,
                "target": "31",
                "prediction": "31",
                "parse_status": "ok",
                "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf", "threshold": 0.5},
            }
        },
    )
    loaded = load_prediction_records(experiment_dir)

    row = loaded.iloc[0]
    assert row["task"] == "corine_level2"
    assert row["text_source"] == "summary"
    assert row["target"] == "31"
    assert row["prediction"] == "31"
    assert row["parse_status"] == "ok"
    assert row["metadata"] == {"model": "Qwen3.6-27B-Q4_0.gguf", "threshold": 0.5}


def test_load_prediction_records_normalizes_list_targets_and_predictions(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "normalized"
    experiment_dir.mkdir()
    path = experiment_dir / "osm_summary_predictions.json"
    _write_json(
        path,
        {"abc": {"pageid": 1, "target": [1, "water"], "prediction": ["wood", 2]}},
    )
    loaded = load_prediction_records(experiment_dir, normalize_targets=True)

    row = loaded.iloc[0]
    assert row["target"] == ["1", "water"]
    assert row["prediction"] == ["wood", "2"]


def test_load_prediction_records_normalizes_scalar_targets_to_string_when_enabled(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "scalar_normalization"
    experiment_dir.mkdir()
    path = experiment_dir / "corine_level2_summary_predictions.json"
    _write_json(
        path,
        {"1": {"pageid": 1, "target": 31, "prediction": 31}},
    )

    loaded = load_prediction_records(experiment_dir, normalize_targets=True)

    row = loaded.iloc[0]
    assert row["target"] == "31"


def test_load_prediction_records_filters_by_text_sources(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "filtered"
    experiment_dir.mkdir()
    _write_json(
        experiment_dir / "corine_level2_summary_predictions.json",
        {"1": {"pageid": "1"}},
    )
    _write_json(
        experiment_dir / "corine_level2_summary_shuffled_predictions.json",
        {"2": {"pageid": "2"}},
    )
    _write_json(
        experiment_dir / "osm_content_shuffled_predictions.json",
        {"3": {"pageid": "3"}},
    )

    loaded = load_prediction_records(
        experiment_dir,
        text_sources={"summary", "content_shuffled"},
    )

    assert set(loaded["text_source"]) == {"summary", "content_shuffled"}
    assert set(loaded["pageid"]) == {"1", "3"}


def test_load_prediction_records_includes_source_parent_dir_when_requested(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "source_parent"
    experiment_dir.mkdir()
    path = experiment_dir / "osm_summary_predictions.json"
    _write_json(path, {"1": {"pageid": 1}})

    loaded = load_prediction_records(experiment_dir, include_source_dir=True)

    assert "source_parent_experiment_dir" in loaded.columns
    assert loaded["source_parent_experiment_dir"].iloc[0] == str(experiment_dir)


def test_load_annotated_prediction_records_adds_model_source_and_optional_model_key(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "annotated"
    experiment_dir.mkdir()
    _write_json(
        experiment_dir / "corine_level2_summary_predictions.json",
        {
            "1": {
                "pageid": 1,
                "target": 31,
                "prediction": 31,
                "metadata": {"model": "explicit-model.gguf"},
            }
        },
    )

    loaded = load_annotated_prediction_records(
        experiment_dir,
        text_sources={"summary"},
        source_group="baseline",
        model_key="qwen",
    )

    row = loaded.iloc[0]
    assert row["target"] == "31"
    assert row["model"] == "explicit-model.gguf"
    assert row["model_key"] == "qwen"
    assert row["source_group"] == "baseline"
    assert row["source_experiment_dir"] == str(experiment_dir)


def test_infer_model_from_metadata_prefers_metadata_and_falls_back_to_directory_name() -> None:
    assert (
        infer_model_from_metadata(
            {"model": "explicit-model.gguf"},
            Path("/tmp/article_text_classification_e2e_with_shuffled_control_v1"),
        )
        == "explicit-model.gguf"
    )
    assert (
        infer_model_from_metadata(
            {"model_repo_id": "unsloth/gemma-4-31B-it-GGUF"},
            Path(
                "/tmp/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
            ),
        )
        == "unsloth/gemma-4-31B-it-GGUF"
    )
    assert (
        infer_model_from_metadata(
            {},
            Path(
                "/tmp/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0",
            ),
        )
        == "gemma-4-31B-it-Q4_0.gguf"
    )
    assert (
        infer_model_from_metadata(
            {}, Path("/tmp/article_text_classification_e2e_with_shuffled_control_v1")
        )
        == "Qwen3.6-27B-Q4_0.gguf"
    )


def test_infer_model_from_metadata_defaults_to_model_repo_id_when_model_is_missing() -> None:
    assert (
        infer_model_from_metadata(
            {"model_repo_id": "unsloth/gemma-4-31B-it-GGUF"},
            Path("/tmp/article_text_classification_e2e_with_shuffled_control_v1"),
        )
        == "unsloth/gemma-4-31B-it-GGUF"
    )


def test_infer_model_from_metadata_respects_explicit_key_order() -> None:
    assert (
        infer_model_from_metadata(
            {"model_repo_id": "unsloth/gemma-4-31B-it-GGUF"},
            Path(
                "/tmp/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
            ),
            metadata_keys=("model",),
        )
        == "gemma-4-31B-it-Q4_0.gguf"
    )
