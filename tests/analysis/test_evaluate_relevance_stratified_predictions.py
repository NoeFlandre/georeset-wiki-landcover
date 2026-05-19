"""Tests for relevance-stratified classification analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from georeset.cli.analysis import evaluate_relevance_stratified_predictions as relevance_eval
from georeset.cli.analysis.evaluate_relevance_stratified_predictions import (
    _write_summary,
    define_evidence_type_subsets,
    define_relevance_and_spatial_subsets,
    define_relevance_subsets,
    load_evidence_metadata,
    main,
)


def _write_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def _write_predictions(
    parent_dir: Path,
    task: str,
    text_source: str,
    records: dict[str, dict[str, object]],
) -> None:
    path = parent_dir / f"{task}_{text_source}_predictions.json"
    _write_json(path, records)


def test_write_summary_empty_rows_uses_shared_markdown_table(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_markdown_table(*, rows: list[dict[str, object]], columns: list[str] | None = None) -> str:
        calls.append({"rows": rows, "columns": columns})
        return "No rows.\n"

    monkeypatch.setattr(relevance_eval, "markdown_table", fake_markdown_table)

    _write_summary(tmp_path, [], [], [])

    assert calls == [{"rows": [], "columns": None}]
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == (
        "# article_text_classification_relevance_stratified_v1\n\nNo rows.\n"
    )


def _make_parent_experiment(
    parent_dir: Path,
    *,
    include_landuse_summary: bool,
    model: str,
    model_repo_id: str | None = None,
) -> None:
    parent_dir.mkdir(exist_ok=True)
    corine_summary = {
        "1": {
            "pageid": 1,
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "2": {
            "pageid": 2,
            "target": "21",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "3": {
            "pageid": 3,
            "target": "31",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "4": {
            "pageid": 4,
            "target": "21",
            "prediction": None,
            "parse_status": "error",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
    }
    corine_summary_shuffled = {
        "1": {
            "pageid": 1,
            "target": "31",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "2": {
            "pageid": 2,
            "target": "21",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "3": {
            "pageid": 3,
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "4": {
            "pageid": 4,
            "target": "21",
            "prediction": None,
            "parse_status": "error",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
    }
    corine_content_shuffled = {
        **{
            key: {"pageid": value["pageid"], "target": value["target"], "prediction": value["prediction"], "parse_status": value["parse_status"], "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})}}
            for key, value in corine_summary_shuffled.items()
        }
    }
    osm_summary = {
        "1": {
            "pageid": 1,
            "target": ["wood"],
            "prediction": ["wood"],
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "2": {
            "pageid": 2,
            "target": ["water"],
            "prediction": [],
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "3": {
            "pageid": 3,
            "target": ["wood", "water"],
            "prediction": ["wood"],
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "4": {
            "pageid": 4,
            "target": ["wood"],
            "prediction": None,
            "parse_status": "error",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
    }
    osm_summary_shuffled = {
        "1": {
            "pageid": 1,
            "target": ["wood"],
            "prediction": [],
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "2": {
            "pageid": 2,
            "target": ["water"],
            "prediction": ["water"],
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "3": {
            "pageid": 3,
            "target": ["wood", "water"],
            "prediction": ["wood", "water"],
            "parse_status": "ok",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
        "4": {
            "pageid": 4,
            "target": ["wood"],
            "prediction": None,
            "parse_status": "error",
            "metadata": {"model": model, **({"model_repo_id": model_repo_id} if model_repo_id else {})},
        },
    }
    for source in [
        "summary",
        "summary_no_place",
        "summary_shuffled",
        "summary_no_place_shuffled",
        "content",
        "content_shuffled",
    ]:
        if source.endswith("_shuffled"):
            if source.startswith("summary"):
                _write_predictions(parent_dir, "corine_level2", source, corine_summary_shuffled)
                _write_predictions(parent_dir, "osm", source, osm_summary_shuffled)
            else:
                _write_predictions(parent_dir, "corine_level2", source, corine_content_shuffled)
                _write_predictions(parent_dir, "osm", source, osm_summary_shuffled)
        else:
            if source == "summary":
                _write_predictions(parent_dir, "corine_level2", source, corine_summary)
                _write_predictions(parent_dir, "osm", source, osm_summary)
            else:
                _write_predictions(parent_dir, "corine_level2", source, corine_summary)
                _write_predictions(parent_dir, "osm", source, osm_summary)

    if include_landuse_summary:
        _write_predictions(parent_dir, "corine_level2", "landuse_evidence_summary", corine_summary)
        _write_predictions(parent_dir, "corine_level2", "landuse_evidence_summary_shuffled", corine_summary_shuffled)
        _write_predictions(parent_dir, "osm", "landuse_evidence_summary", osm_summary)
        _write_predictions(parent_dir, "osm", "landuse_evidence_summary_shuffled", osm_summary_shuffled)


def _make_evidence_metadata(path: Path) -> None:
    _write_json(
        path,
        {
            "1": {
                "pageid": 1,
                "landcover_relevance": "low",
                "uncertainty": "low",
                "evidence_types": ["forest"],
                "evidence_sentences_count": 1,
            },
            "2": {
                "pageid": 2,
                "landcover_relevance": "medium",
                "uncertainty": "medium",
                "evidence_types": ["water", "wetland"],
                "evidence_sentences_count": 2,
            },
            "3": {
                "pageid": 3,
                "landcover_relevance": "high",
                "uncertainty": "high",
                "evidence_types": ["agriculture", "pasture"],
                "evidence_sentences_count": 3,
            },
            "4": {
                "landcover_relevance": "none",
                "uncertainty": "low",
                "evidence_types": ["wetland"],
                "evidence_sentences_count": 1,
            },
        },
    )


def _make_spatial_confidence(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "pageid": "1",
                "point_label_share_250m": 0.9,
                "point_label_share_500m": 0.7,
            },
            {
                "pageid": "2",
                "point_label_share_250m": 0.7,
                "point_label_share_500m": 0.7,
            },
            {
                "pageid": "3",
                "point_label_share_250m": 0.95,
                "point_label_share_500m": 0.95,
            },
            {
                "pageid": "4",
                "point_label_share_250m": 0.95,
                "point_label_share_500m": 0.95,
            },
        ]
    ).to_csv(path, index=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_load_evidence_metadata_normalizes_pageid_to_string(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    _make_evidence_metadata(evidence_path)
    evidence = load_evidence_metadata(evidence_path)
    assert all(isinstance(value, str) for value in evidence["pageid"])
    assert set(evidence["pageid"]) == {"1", "2", "3", "4"}


def test_relevance_subsets_and_spatial_intersections() -> None:
    frame = pd.DataFrame(
        [
            {"pageid": "1", "landcover_relevance": "low", "evidence_sentences_count": 1},
            {"pageid": "2", "landcover_relevance": "medium", "evidence_sentences_count": 2},
            {"pageid": "3", "landcover_relevance": "high", "evidence_sentences_count": 3},
            {"pageid": "4", "landcover_relevance": "none", "evidence_sentences_count": 0},
        ]
    )
    spatial = pd.DataFrame(
        [
            {"pageid": "1", "point_label_share_250m": 0.9},
            {"pageid": "2", "point_label_share_250m": 0.6},
            {"pageid": "3", "point_label_share_250m": 0.95},
            {"pageid": "4", "point_label_share_250m": 0.2},
        ]
    )
    frame_with_spatial = frame.join(spatial.set_index("pageid"), on="pageid")
    relevance = define_relevance_subsets(frame_with_spatial)
    evidence_type = define_evidence_type_subsets(
        frame.assign(evidence_types=[["forest"], ["water"], ["agriculture"], ["bare_ground"]])
    )
    spatial_relevance = define_relevance_and_spatial_subsets(frame.join(spatial.set_index("pageid"), on="pageid"))
    assert set(frame.loc[relevance["relevance_low"], "pageid"]) == {"1"}
    assert set(frame.loc[relevance["relevance_medium_high"], "pageid"]) == {"2", "3"}
    assert set(frame.loc[relevance["point_label_share_250m_ge_0.8_and_relevance_medium_high"], "pageid"]) == {
        "3"
    }
    assert set(frame.loc[evidence_type["evidence_type_forest"], "pageid"]) == {"1"}
    assert set(frame.loc[evidence_type["evidence_type_bare_ground"], "pageid"]) == {"4"}
    assert set(frame.loc[spatial_relevance["all_and_point_label_share_250m_ge_0.8"], "pageid"]) == {
        "1",
        "3",
    }
    assert set(frame.loc[spatial_relevance["relevance_low_and_point_label_share_250m_ge_0.8"], "pageid"]) == {"1"}


@pytest.mark.parametrize(
    ("include_landuse", "expect_optional"),
    [(True, "landuse_evidence_summary"), (False, None)],
)
def test_relevance_stratified_analysis_runs_and_writes_outputs(
    tmp_path: Path,
    include_landuse: bool,
    expect_optional: str | None,
) -> None:
    qwen_parent = tmp_path / "qwen"
    gemma_parent = tmp_path / "gemma"
    _make_parent_experiment(
        qwen_parent, include_landuse_summary=include_landuse, model="Qwen3.6-27B-Q4_0.gguf"
    )
    _make_parent_experiment(
        gemma_parent,
        include_landuse_summary=False,
        model="gemma-4-31B-it-Q4_0.gguf",
        model_repo_id="unsloth/gemma-4-31B-it-GGUF",
    )
    marker = qwen_parent / "README.md"
    marker.write_text("keep-me", encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    _make_evidence_metadata(evidence_path)
    spatial_path = tmp_path / "spatial.csv"
    _make_spatial_confidence(spatial_path)
    output_dir = tmp_path / "out"

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--parent-experiment-dir",
            str(gemma_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "article_text_classification_relevance_stratified_v1",
        ]
    )

    assert marker.read_text(encoding="utf-8") == "keep-me"
    expected_outputs = [
        "overview_by_relevance.csv",
        "overview_by_relevance.md",
        "overview_by_relevance_and_spatial_confidence.csv",
        "overview_by_relevance_and_spatial_confidence.md",
        "overview_by_evidence_type.csv",
        "overview_by_evidence_type.md",
        "shuffled_delta_by_relevance.csv",
        "shuffled_delta_by_relevance.md",
        "majority_baselines_by_relevance.csv",
        "majority_baselines_by_relevance.md",
        "per_class_corine_by_relevance.csv",
        "per_class_corine_by_relevance.md",
        "model_comparison_by_relevance.csv",
        "model_comparison_by_relevance.md",
        "evidence_metadata_distribution.csv",
        "evidence_metadata_distribution.md",
        "manifest.json",
        "summary.md",
    ]
    for filename in expected_outputs:
        assert (output_dir / filename).exists()

    overview = _read_csv(output_dir / "overview_by_relevance.csv")
    summary_row = next(
        row
        for row in overview
        if row["model"] == "Qwen3.6-27B-Q4_0.gguf"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["relevance_subset"] == "all"
    )
    assert summary_row["n"] == "4"
    assert summary_row["n_predicted_ok"] == "3"
    assert summary_row["n_parse_error"] == "1"

    medium_high = next(
        row
        for row in overview
        if row["model"] == "Qwen3.6-27B-Q4_0.gguf"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["relevance_subset"] == "relevance_medium_high"
    )
    assert medium_high["n"] == "2"
    assert medium_high["accuracy"] == "0.0"

    low = next(
        row
        for row in overview
        if row["model"] == "Qwen3.6-27B-Q4_0.gguf"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["relevance_subset"] == "relevance_low"
    )
    assert low["n"] == "1"
    assert low["accuracy"] == "1.0"

    shuffled_delta = _read_csv(output_dir / "shuffled_delta_by_relevance.csv")
    delta = next(
        row
        for row in shuffled_delta
        if row["model"] == "Qwen3.6-27B-Q4_0.gguf"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["relevance_subset"] == "relevance_medium_high"
    )
    assert delta["aligned_score"] == "0.0"
    assert delta["shuffled_score"] == "0.2222222222222222"
    assert delta["delta"] == "-0.2222222222222222"

    dist = _read_csv(output_dir / "evidence_metadata_distribution.csv")
    missing = next(
        row
        for row in dist
        if row["dimension"] == "missing_evidence_records"
        and row["model"] == "Qwen3.6-27B-Q4_0.gguf"
    )
    assert missing["count"] == "0"

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_llm_rerun"] is True
    assert manifest["experiment_id"] == "article_text_classification_relevance_stratified_v1"
    assert manifest["input_paths"]["evidence_metadata_path"] == str(evidence_path)
    assert len(manifest["source_parent_experiment_dirs"]) == 2

    if expect_optional:
        overview_sources = {row["text_source"] for row in overview if row["model"] == "Qwen3.6-27B-Q4_0.gguf"}
        assert expect_optional in overview_sources
        gemma_sources = {
            row["text_source"] for row in overview if row["model"] == "gemma-4-31B-it-Q4_0.gguf"
        }
        assert expect_optional not in gemma_sources
    else:
        overview_sources = {row["text_source"] for row in overview if row["model"] == "Qwen3.6-27B-Q4_0.gguf"}
        assert "landuse_evidence_summary" not in overview_sources


def test_metrics_are_recomputed_per_subset_and_majority_baseline_is_not_global(tmp_path: Path) -> None:
    qwen_parent = tmp_path / "qwen"
    _make_parent_experiment(qwen_parent, include_landuse_summary=False, model="Qwen3.6-27B-Q4_0.gguf")
    evidence_path = tmp_path / "evidence.json"
    _make_evidence_metadata(evidence_path)
    spatial_path = tmp_path / "spatial.csv"
    _make_spatial_confidence(spatial_path)
    output_dir = tmp_path / "out"

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    overview = _read_csv(output_dir / "overview_by_relevance.csv")
    all_row = next(
        row for row in overview if row["relevance_subset"] == "relevance_low_medium_high" and row["text_source"] == "summary"
    )
    low_row = next(
        row for row in overview if row["relevance_subset"] == "relevance_low" and row["text_source"] == "summary"
    )
    assert all_row["accuracy"] == "0.3333333333333333"
    assert low_row["accuracy"] == "1.0"
    assert all_row["majority_accuracy"] != low_row["majority_accuracy"]

    majority_rows = _read_csv(output_dir / "majority_baselines_by_relevance.csv")
    majority_low = next(
        row
        for row in majority_rows
        if row["relevance_subset"] == "relevance_low"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
    )
    majority_all = next(
        row
        for row in majority_rows
        if row["relevance_subset"] == "relevance_low_medium_high"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
    )
    assert majority_low["majority_accuracy"] == "1.0"
    assert majority_all["majority_accuracy"] != "1.0"
