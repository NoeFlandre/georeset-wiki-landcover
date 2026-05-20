"""Tests for article supervision quality-score evaluation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from georeset.analysis.list_normalization import normalize_string_list
from georeset.cli.analysis.evaluate_supervision_quality_score import (
    classify_quality_bin,
    compute_quality_row,
    main,
)

TEXT_SOURCES = [
    "summary",
    "summary_shuffled",
    "summary_no_place",
    "summary_no_place_shuffled",
    "content",
    "content_shuffled",
]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_predictions(path: Path, records: dict[str, dict[str, object]]) -> None:
    _write_json(path, records)


def _build_parent_experiment(
    path: Path,
    *,
    model: str,
    model_repo_id: str | None = None,
) -> None:
    path.mkdir()

    corine_base = {
        "1": {
            "pageid": "1",
            "title": "A",
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "2": {
            "pageid": 2,
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "3": {
            "pageid": "3",
            "title": "C",
            "target": "32",
            "prediction": "23",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "4": {
            "pageid": "4",
            "target": "51",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
    }
    corine_shuffled = {
        "1": {
            "pageid": "1",
            "title": "A",
            "target": "31",
            "prediction": "21",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "2": {
            "pageid": 2,
            "title": "B",
            "target": "31",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "3": {
            "pageid": "3",
            "target": "32",
            "prediction": "31",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "4": {
            "pageid": "4",
            "target": "51",
            "prediction": "51",
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
    }
    osm_base = {
        "1": {
            "pageid": "1",
            "target": ["wood"],
            "prediction": ["wood"],
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "2": {
            "pageid": 2,
            "target": ["wood", "water"],
            "prediction": ["water"],
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "3": {
            "pageid": "3",
            "target": ["water"],
            "prediction": ["water"],
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "4": {
            "pageid": "4",
            "target": ["natural"],
            "prediction": [],
            "parse_status": "error",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
    }
    osm_shuffled = {
        "1": {
            "pageid": "1",
            "target": ["wood"],
            "prediction": ["water"],
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "2": {
            "pageid": 2,
            "target": ["wood", "water"],
            "prediction": ["wood"],
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "3": {
            "pageid": "3",
            "target": ["water"],
            "prediction": ["wood", "water"],
            "parse_status": "ok",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
        "4": {
            "pageid": "4",
            "target": ["natural"],
            "prediction": [],
            "parse_status": "error",
            "metadata": {
                "model": model,
                **({"model_repo_id": model_repo_id} if model_repo_id else {}),
            },
        },
    }

    for source in TEXT_SOURCES:
        if "shuffled" in source:
            _write_predictions(path / f"corine_level2_{source}_predictions.json", corine_shuffled)
            _write_predictions(path / f"osm_{source}_predictions.json", osm_shuffled)
        else:
            _write_predictions(path / f"corine_level2_{source}_predictions.json", corine_base)
            _write_predictions(path / f"osm_{source}_predictions.json", osm_base)


def _build_evidence_metadata(path: Path) -> None:
    _write_json(
        path,
        {
            "1": {
                "pageid": 1,
                "landcover_relevance": "high",
                "uncertainty": "low",
                "evidence_types": ["forest"],
                "evidence_sentences_count": 2,
                "landuse_evidence_summary_char_count": 30,
            },
            "2": {
                "pageid": "2",
                "landcover_relevance": "medium",
                "uncertainty": "low",
                "evidence_types": ["water"],
                "evidence_sentences_count": 1,
                "landuse_evidence_summary_char_count": 50,
            },
            "3": {
                "pageid": "3",
                "landcover_relevance": "high",
                "uncertainty": "medium",
                "evidence_types": ["wetland"],
                "evidence_sentences_count": 2,
                "landuse_evidence_summary_char_count": 70,
            },
            "4": {
                "pageid": "4",
                "landcover_relevance": "none",
                "uncertainty": "high",
                "evidence_types": [],
                "evidence_sentences_count": 0,
                "landuse_evidence_summary_char_count": 10,
            },
        },
    )


def _build_spatial_confidence(path: Path) -> None:
    pd.DataFrame(
        [
            {"pageid": "1", "point_label": "31", "point_label_share_250m": 0.95},
            {"pageid": "2", "point_label": "31", "point_label_share_250m": 0.82},
            {"pageid": "3", "point_label": "32", "point_label_share_250m": 0.52},
        ]
    ).to_csv(path, index=False)


def _build_wiki_articles(path: Path) -> None:
    _write_json(
        path,
        [
            {"pageid": "1", "title": "A", "lat": 46.0, "lon": 7.0},
            {"pageid": 2, "title": "B", "lat": 47.0, "lon": 7.1},
            {"pageid": "3", "title": "C", "lat": 47.1, "lon": 7.2},
            {"pageid": 4, "title": "D", "lat": 47.2, "lon": 7.3},
        ],
    )


def _build_article_type_metadata(path: Path) -> None:
    _write_json(
        path,
        {
            "1": {
                "pageid": 1,
                "primary_article_type": "agriculture_or_vineyard",
                "candidate_article_types": ["agriculture_or_vineyard"],
                "title": "A",
            },
            "2": {
                "pageid": "2",
                "primary_article_type": "natural_landscape",
                "candidate_article_types": ["natural_landscape"],
                "title": "B",
            },
            "3": {
                "pageid": "3",
                "primary_article_type": "water_feature",
                "candidate_article_types": ["water_feature"],
                "title": "C",
            },
            # pageid 4 missing on purpose (article type missing)
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_score_components_and_conservative_missing_values() -> None:
    row = compute_quality_row(
        {
            "landcover_relevance": "unknown",
            "uncertainty": "unknown",
            "point_label_share_250m": None,
            "evidence_sentences_count": None,
            "primary_article_type": None,
        }
    )
    assert row["relevance_score"] == 0
    assert row["spatial_score"] == 0
    assert row["evidence_density_score"] == 0
    assert row["article_type_score"] == 0
    assert row["uncertainty_penalty"] == 1
    assert row["quality_score"] == -1
    assert row["quality_bin"] == "quality_low"
    assert row["recommended_use"] == "exclude"


def test_quality_bin_edges() -> None:
    assert classify_quality_bin(2.999) == "quality_low"
    assert classify_quality_bin(3.0) == "quality_medium"
    assert classify_quality_bin(4.999) == "quality_medium"
    assert classify_quality_bin(5.0) == "quality_high"
    assert classify_quality_bin(6.999) == "quality_high"
    assert classify_quality_bin(7.0) == "quality_very_high"


def test_local_list_normalizer_drops_missing_items_and_parses_json_null() -> None:
    assert normalize_string_list(["forest", None, "", "   ", math.nan]) == ["forest"]
    assert normalize_string_list('["water", null, ""]') == ["water"]


def test_recommended_use_precedence() -> None:
    very_high_row = compute_quality_row(
        {
            "landcover_relevance": "high",
            "uncertainty": "low",
            "point_label_share_250m": 0.95,
            "evidence_sentences_count": 2,
            "primary_article_type": "natural_landscape",
        }
    )
    assert very_high_row["quality_bin"] == "quality_very_high"
    assert very_high_row["recommended_use"] == "use_for_evaluation_only"

    high_with_uncertainty_blocked = compute_quality_row(
        {
            "landcover_relevance": "high",
            "uncertainty": "high",
            "point_label_share_250m": 0.95,
            "evidence_sentences_count": 2,
            "primary_article_type": "natural_landscape",
        }
    )
    assert high_with_uncertainty_blocked["recommended_use"] == "exclude"

    quality_high_training = compute_quality_row(
        {
            "landcover_relevance": "medium",
            "uncertainty": "low",
            "point_label_share_250m": 0.81,
            "evidence_sentences_count": 2,
            "primary_article_type": "settlement_or_administrative",
        }
    )
    assert quality_high_training["quality_bin"] in {"quality_high", "quality_very_high"}
    assert quality_high_training["recommended_use"] == "use_for_training"

    inspect_manual_if_missing_article_type = compute_quality_row(
        {
            "landcover_relevance": "low",
            "uncertainty": "low",
            "point_label_share_250m": 0.70,
            "evidence_sentences_count": 0,
            "primary_article_type": None,
        }
    )
    assert inspect_manual_if_missing_article_type["quality_bin"] in {
        "quality_medium",
        "quality_high",
        "quality_very_high",
    }
    assert inspect_manual_if_missing_article_type["recommended_use"] == "inspect_manually"


def test_string_pageid_joins_and_candidate_output_fields(tmp_path: Path) -> None:
    qwen_parent = tmp_path / "qwen"
    gemma_parent = tmp_path / "gemma"
    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    _build_parent_experiment(
        gemma_parent,
        model="gemma-4-31B-it-Q4_0.gguf",
        model_repo_id="unsloth/gemma-4-31B-it-GGUF",
    )

    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    wiki_path = tmp_path / "wiki_articles.json"
    article_type_path = tmp_path / "article_type_metadata.json"

    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_wiki_articles(wiki_path)
    _write_json(
        article_type_path,
        {
            "1": {
                "pageid": 1,
                "primary_article_type": "agriculture_or_vineyard",
                "candidate_article_types": ["agriculture_or_vineyard"],
                "title": "TYPE_A",
            },
            "2": {
                "pageid": "2",
                "primary_article_type": "natural_landscape",
                "candidate_article_types": ["natural_landscape"],
                "title": "WRONG_TITLE",
            },
            "3": {
                "pageid": "3",
                "primary_article_type": "water_feature",
                "candidate_article_types": ["water_feature"],
                "title": "TYPE_C",
            },
        },
    )

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--parent-experiment-dir",
            str(gemma_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--wiki-articles-path",
            str(wiki_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--article-type-metadata-path",
            str(article_type_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    candidate_rows = _read_csv(output_dir / "candidate_training_pairs_by_quality.csv")
    assert {row["pageid"] for row in candidate_rows} == {"1", "2", "3", "4"}
    row_2 = next(row for row in candidate_rows if row["pageid"] == "2")
    assert row_2["title"] == "B"
    first_row = candidate_rows[0]
    expected_columns = {
        "pageid",
        "title",
        "lat",
        "lon",
        "corine_label",
        "osm_labels",
        "primary_article_type",
        "candidate_article_types",
        "landcover_relevance",
        "uncertainty",
        "evidence_types",
        "evidence_sentences_count",
        "landuse_evidence_summary_char_count",
        "point_label_share_250m",
        "point_label_share_500m",
        "dominant_matches_point_label_250m",
        "relevance_score",
        "spatial_score",
        "evidence_density_score",
        "article_type_score",
        "uncertainty_penalty",
        "quality_score",
        "quality_bin",
        "recommended_use",
    }
    assert expected_columns.issubset(set(first_row))


def test_quality_score_output_normalizes_list_metadata_without_fake_missing_values(
    tmp_path: Path,
) -> None:
    qwen_parent = tmp_path / "qwen"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    wiki_path = tmp_path / "wiki_articles.json"
    article_type_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    _build_spatial_confidence(spatial_path)
    _build_wiki_articles(wiki_path)
    _write_json(
        evidence_path,
        {
            "1": {
                "pageid": "1",
                "landcover_relevance": "high",
                "uncertainty": "low",
                "evidence_types": ["forest", None, "", "   ", math.nan],
                "evidence_sentences_count": 1,
            },
            "2": {
                "pageid": "2",
                "landcover_relevance": "medium",
                "uncertainty": "low",
                "evidence_types": '["water", null, ""]',
                "evidence_sentences_count": 1,
            },
        },
    )
    _write_json(
        article_type_path,
        {
            "1": {
                "pageid": "1",
                "primary_article_type": "natural_landscape",
                "candidate_article_types": ["natural_landscape", None, "", math.nan],
            },
            "2": {
                "pageid": "2",
                "primary_article_type": "water_feature",
                "candidate_article_types": '["water_feature", null, ""]',
            },
        },
    )

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--wiki-articles-path",
            str(wiki_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--article-type-metadata-path",
            str(article_type_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    quality_rows = _read_csv(output_dir / "quality_scores.csv")
    row_1 = next(row for row in quality_rows if row["pageid"] == "1")
    row_2 = next(row for row in quality_rows if row["pageid"] == "2")

    assert row_1["evidence_types"] == "['forest']"
    assert row_1["candidate_article_types"] == "['natural_landscape']"
    assert row_2["evidence_types"] == "['water']"
    assert row_2["candidate_article_types"] == "['water_feature']"


def test_manifest_contains_counts_and_fields(tmp_path: Path) -> None:
    qwen_parent = tmp_path / "qwen"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    wiki_path = tmp_path / "wiki_articles.json"
    article_type_path = tmp_path / "article_type.json"

    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_wiki_articles(wiki_path)
    _build_article_type_metadata(article_type_path)

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--wiki-articles-path",
            str(wiki_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--article-type-metadata-path",
            str(article_type_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "article_text_supervision_quality_score_v1"
    assert manifest["no_llm_rerun"] is True
    assert manifest["input_paths"]["evidence_metadata_path"] == str(evidence_path)
    assert manifest["input_paths"]["wiki_articles_path"] == str(wiki_path)
    assert manifest["input_paths"]["spatial_confidence_path"] == str(spatial_path)
    assert manifest["number_of_articles_scored"] == 4
    assert "models" in manifest
    assert manifest["missing_metadata_counts"]["evidence_metadata_missing"] == 0


@pytest.mark.parametrize(
    "filename",
    [
        "quality_scores.csv",
        "quality_scores.md",
        "candidate_training_pairs_by_quality.csv",
        "candidate_training_pairs_by_quality.md",
        "quality_bin_distribution.csv",
        "quality_bin_distribution.md",
        "recommended_use_distribution.csv",
        "recommended_use_distribution.md",
        "overview_by_quality_bin.csv",
        "overview_by_quality_bin.md",
        "overview_by_quality_and_spatial.csv",
        "overview_by_quality_and_spatial.md",
        "overview_comparison_simple_filters.csv",
        "overview_comparison_simple_filters.md",
        "shuffled_delta_by_quality_bin.csv",
        "shuffled_delta_by_quality_bin.md",
        "majority_baselines_by_quality_bin.csv",
        "majority_baselines_by_quality_bin.md",
        "per_class_corine_by_quality_bin.csv",
        "per_class_corine_by_quality_bin.md",
        "model_comparison_by_quality_bin.csv",
        "model_comparison_by_quality_bin.md",
        "manifest.json",
        "summary.md",
    ],
)
def test_required_output_files_exist(tmp_path: Path, filename: str) -> None:
    qwen_parent = tmp_path / "qwen"
    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    output_dir = tmp_path / "out"
    _build_evidence_metadata(tmp_path / "evidence.json")
    _build_spatial_confidence(tmp_path / "spatial.csv")
    _build_wiki_articles(tmp_path / "wiki_articles.json")
    _build_article_type_metadata(tmp_path / "article_type_metadata.json")

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(tmp_path / "evidence.json"),
            "--wiki-articles-path",
            str(tmp_path / "wiki_articles.json"),
            "--spatial-confidence-path",
            str(tmp_path / "spatial.csv"),
            "--article-type-metadata-path",
            str(tmp_path / "article_type_metadata.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert (output_dir / filename).exists()


def test_metrics_include_corine_full_labels_and_osm_multi_metrics(tmp_path: Path) -> None:
    qwen_parent = tmp_path / "qwen"
    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    wiki_path = tmp_path / "wiki_articles.json"
    article_type_path = tmp_path / "article_type_metadata.json"
    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_wiki_articles(wiki_path)
    _build_article_type_metadata(article_type_path)

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--wiki-articles-path",
            str(wiki_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--article-type-metadata-path",
            str(article_type_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    per_class = _read_csv(output_dir / "per_class_corine_by_quality_bin.csv")
    labels = {row["label"] for row in per_class}
    assert len(labels) >= 8
    assert "31" in labels

    osm_rows = [
        row for row in _read_csv(output_dir / "overview_by_quality_bin.csv") if row["task"] == "osm"
    ]
    assert osm_rows
    for metric_name in ["jaccard", "micro_f1", "macro_f1"]:
        assert metric_name in osm_rows[0]


def test_subset_metrics_and_shuffled_deltas(tmp_path: Path) -> None:
    qwen_parent = tmp_path / "qwen"
    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    wiki_path = tmp_path / "wiki_articles.json"
    article_type_path = tmp_path / "article_type_metadata.json"

    _build_evidence_metadata(evidence_path)
    _build_spatial_confidence(spatial_path)
    _build_wiki_articles(wiki_path)
    _build_article_type_metadata(article_type_path)

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--wiki-articles-path",
            str(wiki_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--article-type-metadata-path",
            str(article_type_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    quality_rows = _read_csv(output_dir / "overview_by_quality_bin.csv")
    all_row = next(
        row for row in quality_rows if row["subset"] == "all" and row["task"] == "corine_level2"
    )
    medium_row = next(
        row
        for row in quality_rows
        if row["subset"] == "quality_medium" and row["task"] == "corine_level2"
    )
    assert all_row["n"] != medium_row["n"]
    assert float(all_row["balanced_accuracy"]) != float(medium_row["balanced_accuracy"])

    deltas = _read_csv(output_dir / "shuffled_delta_by_quality_bin.csv")
    delta_row = next(
        row
        for row in deltas
        if row["subset"] == "all"
        and row["task"] == "corine_level2"
        and row["text_source"] == "summary"
    )
    assert float(delta_row["delta"]) == float(delta_row["aligned_score"]) - float(
        delta_row["shuffled_score"]
    )


def test_missing_metadata_counts_and_no_parent_dir_mutation(tmp_path: Path) -> None:
    qwen_parent = tmp_path / "qwen"
    output_dir = tmp_path / "out"
    evidence_path = tmp_path / "evidence.json"
    spatial_path = tmp_path / "spatial.csv"
    wiki_path = tmp_path / "wiki_articles.json"
    article_type_path = tmp_path / "article_type_metadata.json"

    _build_parent_experiment(qwen_parent, model="Qwen3.6-27B-Q4_0.gguf")
    _build_spatial_confidence(spatial_path)

    # pageid 2 missing in evidence metadata; pageid 4 missing in spatial confidence.
    _write_json(
        evidence_path,
        {
            "1": {
                "pageid": 1,
                "landcover_relevance": "high",
                "uncertainty": "low",
                "evidence_types": ["forest"],
                "evidence_sentences_count": 2,
                "landuse_evidence_summary_char_count": 10,
            },
            "2": {
                "pageid": 2,
                "landcover_relevance": "medium",
                "uncertainty": "low",
                "evidence_types": ["forest"],
                "evidence_sentences_count": 1,
                "landuse_evidence_summary_char_count": 10,
            },
            "3": {
                "pageid": 3,
                "landcover_relevance": "high",
                "uncertainty": "medium",
                "evidence_types": ["forest"],
                "evidence_sentences_count": 2,
                "landuse_evidence_summary_char_count": 10,
            },
            # pageid 4 intentionally missing in evidence metadata
        },
    )
    _build_wiki_articles(wiki_path)
    # only pageid 1 has article-type metadata; others are missing.
    _write_json(
        article_type_path,
        {
            "1": {
                "pageid": 1,
                "primary_article_type": "water_feature",
                "candidate_article_types": ["water_feature"],
            }
        },
    )
    # remove one row from spatial to create missing spatial coverage for pageid 4
    pd.DataFrame(
        [
            {"pageid": "1", "point_label": "31", "point_label_share_250m": 0.95},
            {"pageid": 2, "point_label": "31", "point_label_share_250m": 0.82},
            {"pageid": "3", "point_label": "32", "point_label_share_250m": 0.52},
        ]
    ).to_csv(spatial_path, index=False)

    marker = qwen_parent / "README.md"
    marker.write_text("frozen", encoding="utf-8")

    main(
        [
            "--parent-experiment-dir",
            str(qwen_parent),
            "--evidence-metadata-path",
            str(evidence_path),
            "--wiki-articles-path",
            str(wiki_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--article-type-metadata-path",
            str(article_type_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["missing_metadata_counts"]
    assert counts["evidence_metadata_missing"] == 1
    assert counts["spatial_metadata_missing"] == 1
    assert counts["article_type_metadata_missing"] == 3

    assert marker.read_text(encoding="utf-8") == "frozen"
    assert not any(path.name.startswith("quality_scores_") for path in qwen_parent.iterdir())
