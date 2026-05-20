"""Tests for article-type + relevance + spatial reevaluation."""

import csv
import json
from pathlib import Path

import pandas as pd

from georeset_wiki_landcover.cli.analysis.evaluate_article_type_relevance_stratified import main


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_predictions(
    parent_dir: Path,
    task: str,
    text_source: str,
    records: dict[str, dict[str, object]],
) -> None:
    _write_json(parent_dir / f"{task}_{text_source}_predictions.json", records)


def _build_parent_experiment(path: Path) -> None:
    path.mkdir()
    corine = {
        "1": {
            "pageid": "1",
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "2": {
            "pageid": 2,
            "target": "21",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "3": {
            "pageid": "3",
            "target": "31",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
    }
    corine_shuffled = {
        "1": {
            "pageid": "1",
            "target": "31",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "2": {
            "pageid": "2",
            "target": "21",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "3": {
            "pageid": "3",
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
    }
    osm = {
        "1": {
            "pageid": "1",
            "target": ["wood"],
            "prediction": ["wood"],
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "2": {
            "pageid": "2",
            "target": ["water"],
            "prediction": [],
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "3": {
            "pageid": "3",
            "target": ["wood", "water"],
            "prediction": ["wood"],
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
    }
    osm_shuffled = {
        "1": {
            "pageid": "1",
            "target": ["wood"],
            "prediction": [],
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "2": {
            "pageid": "2",
            "target": ["water"],
            "prediction": ["water"],
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
        "3": {
            "pageid": 3,
            "target": ["wood", "water"],
            "prediction": ["wood", "water"],
            "parse_status": "ok",
            "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
        },
    }

    for source in [
        "summary",
        "summary_no_place",
        "content",
        "summary_shuffled",
        "summary_no_place_shuffled",
        "content_shuffled",
    ]:
        if source.endswith("shuffled"):
            _write_predictions(path, "corine_level2", source, corine_shuffled)
            _write_predictions(path, "osm", source, osm_shuffled)
        else:
            _write_predictions(path, "corine_level2", source, corine)
            _write_predictions(path, "osm", source, osm)


def _build_evidence_metadata(path: Path, *, missing_pageids: set[str] | None = None) -> None:
    evidence = {
        "1": {
            "pageid": 1,
            "landcover_relevance": "high",
            "uncertainty": "low",
            "evidence_types": ["forest"],
            "evidence_sentences_count": 1,
        },
        "2": {
            "pageid": "2",
            "landcover_relevance": "low",
            "uncertainty": "medium",
            "evidence_types": ["water"],
            "evidence_sentences_count": 1,
        },
        "3": {
            "pageid": "3",
            "landcover_relevance": "medium",
            "uncertainty": "high",
            "evidence_types": ["urban_or_artificial"],
            "evidence_sentences_count": 2,
        },
    }
    if missing_pageids:
        for pageid in missing_pageids:
            evidence.pop(str(pageid), None)
            evidence.pop(str(int(pageid)) if pageid.isdigit() else pageid, None)
    _write_json(path, evidence)


def _build_spatial_confidence(path: Path) -> None:
    pd.DataFrame(
        [
            {"pageid": "1", "point_label_share_250m": 0.9},
            {"pageid": "2", "point_label_share_250m": 0.7},
            {"pageid": "3", "point_label_share_250m": 0.95},
        ]
    ).to_csv(path, index=False)


def _build_article_type_metadata(path: Path) -> None:
    _write_json(
        path,
        {
            "1": {
                "pageid": 1,
                "title": "Article One",
                "primary_article_type": "water_feature",
                "candidate_article_types": ["water_feature"],
                "matched_categories": ["rivière"],
                "matched_rules": ["rivière"],
                "all_categories_count": 1,
                "has_categories": True,
            },
            "2": {
                "pageid": 2,
                "title": "Article Two",
                "primary_article_type": "natural_landscape",
                "candidate_article_types": ["natural_landscape"],
                "matched_categories": ["forêt"],
                "matched_rules": ["forêt"],
                "all_categories_count": 1,
                "has_categories": True,
            },
            "3": {
                "pageid": 3,
                "title": "Article Three",
                "primary_article_type": "other_or_unclear",
                "candidate_article_types": ["other_or_unclear"],
                "matched_categories": [],
                "matched_rules": [],
                "all_categories_count": 0,
                "has_categories": False,
            },
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_evaluate_article_type_relevance_stratified_joins_by_pageid_and_outputs_files(
    tmp_path: Path,
) -> None:
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    article_type_metadata_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(parent_dir)
    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_article_type_metadata(article_type_metadata_path)

    marker = parent_dir / "README.md"
    marker.write_text("frozen", encoding="utf-8")

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--article-type-metadata-path",
            str(article_type_metadata_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert marker.read_text(encoding="utf-8") == "frozen"
    for filename in [
        "article_type_assignments.csv",
        "article_type_assignments.md",
        "article_type_assignment_audit_sample.csv",
        "article_type_assignment_audit_sample.md",
        "overview_by_article_type.csv",
        "overview_by_article_type.md",
        "overview_by_article_type_relevance_spatial.csv",
        "overview_by_article_type_relevance_spatial.md",
        "shuffled_delta_by_article_type.csv",
        "shuffled_delta_by_article_type.md",
        "majority_baselines_by_article_type.csv",
        "majority_baselines_by_article_type.md",
        "per_class_corine_by_article_type.csv",
        "per_class_corine_by_article_type.md",
        "article_type_distribution.csv",
        "article_type_distribution.md",
        "manifest.json",
        "summary.md",
    ]:
        assert (output_dir / filename).exists()

    assignments = _read_csv(output_dir / "article_type_assignments.csv")
    assert {row["primary_article_type"] for row in assignments} == {
        "water_feature",
        "natural_landscape",
        "other_or_unclear",
    }
    assert all(row["pageid"] in {"1", "2", "3"} for row in assignments)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_llm_rerun"] is True
    assert manifest["input_paths"]["article_type_metadata_path"] == str(article_type_metadata_path)
    assert manifest["n_prediction_records_loaded"] == 36
    assert manifest["n_unique_prediction_pageids"] == 3
    assert manifest["n_article_type_metadata_records"] == 3
    assert manifest["n_unique_article_type_pageids"] == 3
    assert manifest["n_predictions_missing_article_type_metadata"] == 0
    assert manifest["n_predictions_missing_evidence_metadata"] == 0
    assert manifest["n_predictions_missing_spatial_confidence"] == 0


def test_manifest_counts_missing_evidence_metadata(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    article_type_metadata_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(parent_dir)
    _build_evidence_metadata(evidence_path, missing_pageids={"2"})
    _build_spatial_confidence(spatial_path)
    _build_article_type_metadata(article_type_metadata_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--article-type-metadata-path",
            str(article_type_metadata_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_predictions_missing_evidence_metadata"] == 12
    assert manifest["n_predictions_missing_article_type_metadata"] == 0
    assert manifest["n_predictions_missing_spatial_confidence"] == 0
    assert manifest["n_prediction_records_loaded"] == 36


def test_evaluation_metrics_and_deltas_are_recomputed_per_subset(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    article_type_metadata_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(parent_dir)
    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_article_type_metadata(article_type_metadata_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--article-type-metadata-path",
            str(article_type_metadata_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "article_text_classification_article_type_relevance_stratified_v1",
        ]
    )

    overview = _read_csv(output_dir / "overview_by_article_type.csv")
    water_row = next(
        row
        for row in overview
        if row["subset"] == "article_type:water_feature" and row["task"] == "corine_level2"
    )
    natural_row = next(
        row
        for row in overview
        if row["subset"] == "article_type:natural_landscape" and row["task"] == "corine_level2"
    )
    assert water_row["n"] == "1"
    assert natural_row["n"] == "1"
    assert float(water_row["accuracy"]) == 1.0
    assert float(natural_row["accuracy"]) == 0.0

    delta = next(
        row
        for row in _read_csv(output_dir / "shuffled_delta_by_article_type.csv")
        if row["subset"] == "article_type:water_feature"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["shuffled_text_source"] == "summary_shuffled"
    )
    assert float(delta["delta"]) == float(delta["aligned_score"]) - float(delta["shuffled_score"])

    majorities = _read_csv(output_dir / "majority_baselines_by_article_type.csv")
    water_majority = next(
        row
        for row in majorities
        if row["subset"] == "article_type:water_feature"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
    )
    natural_majority = next(
        row
        for row in majorities
        if row["subset"] == "article_type:natural_landscape"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
    )
    assert water_majority["majority_accuracy"] == "1.0"
    assert natural_majority["majority_accuracy"] == "1.0"

    # relevance + spatial subset family should include both dimensions in its label
    overview_rs = _read_csv(output_dir / "overview_by_article_type_relevance_spatial.csv")
    rs_subsets = {row["subset"] for row in overview_rs}
    assert (
        "article_type:water_feature|relevance:high|spatial:point_label_share_250m_ge_0.8"
        in rs_subsets
    )
    assert (
        "article_type:natural_landscape|relevance:low|spatial:point_label_share_250m_ge_0.8"
        in rs_subsets
    )


def test_audit_sample_is_deterministic_and_has_required_columns(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    article_type_metadata_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(parent_dir)
    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_article_type_metadata(article_type_metadata_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--article-type-metadata-path",
            str(article_type_metadata_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    sample = _read_csv(output_dir / "article_type_assignment_audit_sample.csv")
    assert list(sample[0].keys()) == [
        "pageid",
        "title",
        "primary_article_type",
        "candidate_article_types",
        "matched_categories",
        "matched_rules",
        "all_categories_count",
        "has_categories",
    ]
    assert [row["primary_article_type"] for row in sample] == sorted(
        ["natural_landscape", "other_or_unclear", "water_feature"]
    )


def test_parent_dirs_not_mutated(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    article_type_metadata_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(parent_dir)
    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_article_type_metadata(article_type_metadata_path)

    marker = parent_dir / "README.md"
    marker.write_text("frozen", encoding="utf-8")

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--article-type-metadata-path",
            str(article_type_metadata_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "frozen"
    # no new article-type files are written into source experiment directories
    assert not any(path.name.startswith("article_type_") for path in parent_dir.iterdir())
