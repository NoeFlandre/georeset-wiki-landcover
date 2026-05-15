"""Compute analysis-only quality scores for supervised article text predictions."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.analysis.article_type_metadata_loading import load_article_type_metadata
from georeset.analysis.evaluation_metrics import (
    compute_multilabel_subset_metrics,
    compute_single_label_subset_metrics,
)
from georeset.analysis.evidence_metadata_loading import load_evidence_metadata
from georeset.analysis.label_universe import label_universe
from georeset.analysis.list_normalization import normalize_string_list
from georeset.analysis.prediction_loading import (
    infer_model_for_records,
    load_prediction_records,
)
from georeset.analysis.shuffled_deltas import primary_metric_name
from georeset.analysis.spatial_confidence_loading import load_spatial_confidence
from georeset.classification.text_sources import shuffled_text_source_pairs
from georeset.utils.json_io import (
    read_json_file,
    write_dict_rows_csv_atomic,
    write_dict_rows_markdown_atomic,
    write_json_atomic,
    write_text_atomic,
)

DEFAULT_PARENT_DIRS = [
    Path("data/experiments/article_text_classification_e2e_with_shuffled_control_v1"),
    Path("data/experiments/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"),
]
DEFAULT_WIKI_ARTICLES_PATH = Path("data/wiki/wiki_articles.json")
DEFAULT_EVIDENCE_METADATA_PATH = Path("data/wiki/article_landuse_evidence_summaries.json")
DEFAULT_SPATIAL_CONFIDENCE_PATH = Path(
    "data/experiments/corine_spatial_confidence_v1/spatial_confidence.csv"
)
DEFAULT_ARTICLE_TYPE_METADATA_PATH = Path(
    "data/experiments/article_text_classification_article_type_relevance_stratified_v1/article_type_assignments.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/experiments/article_text_supervision_quality_score_v1")
DEFAULT_EXPERIMENT_ID = "article_text_supervision_quality_score_v1"

TEXT_SOURCES = [
    "summary",
    "summary_shuffled",
    "summary_no_place",
    "summary_no_place_shuffled",
    "content",
    "content_shuffled",
]
QUALITY_BINS: dict[str, Callable[[float], bool]] = {
    "quality_low": lambda score: score < 3.0,
    "quality_medium": lambda score: 3.0 <= score < 5.0,
    "quality_high": lambda score: 5.0 <= score < 7.0,
    "quality_very_high": lambda score: score >= 7.0,
}
ARTICLE_TYPE_TRAIN_SCORE = {
    "agriculture_or_vineyard": 2,
    "natural_landscape": 2,
    "water_feature": 2,
    "settlement_or_administrative": 1,
    "built_or_cultural_site": 1,
}
ARTICLE_TYPE_HIGH_PRIOR = {
    "agriculture_or_vineyard",
    "natural_landscape",
    "water_feature",
}
RECOMMENDED_USE_PRECEDENCE = [
    "exclude if quality_low OR landcover_relevance == none OR uncertainty == high",
    "use_for_evaluation_only if quality_very_high AND "
    "point_label_share_250m >= 0.9 AND landcover_relevance == high",
    "use_for_training if quality_bin in {quality_high, quality_very_high} "
    "AND point_label_share_250m >= 0.8 AND landcover_relevance in {medium, high}",
    "inspect_manually if quality_medium OR missing article type OR uncertainty == medium",
]

QUALITY_SCORE_FORMULA = (
    "relevance_score + spatial_score + evidence_density_score + "
    "article_type_score - uncertainty_penalty"
)

OUTPUT_METRIC_COLUMNS = [
    "subset",
    "model",
    "task",
    "text_source",
    "n",
    "n_predicted_ok",
    "n_parse_error",
    "coverage",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "weighted_f1",
    "jaccard",
    "hamming_loss",
    "majority_accuracy",
    "majority_balanced_accuracy",
    "majority_macro_f1",
    "majority_labelset_exact_match_accuracy",
    "empty_set_exact_match_accuracy",
    "exact_match_accuracy",
]

QUALITY_SCORE_COLUMNS = [
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
]


def _normalize_optional_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def classify_quality_bin(score: float) -> str:
    for key, check in QUALITY_BINS.items():
        if check(score):
            return key
    raise ValueError(f"Unexpected score: {score}")


def compute_quality_row(record: Mapping[str, Any]) -> dict[str, Any]:
    relevance = _normalize_optional_text(record.get("landcover_relevance"))
    uncertainty = _normalize_optional_text(record.get("uncertainty"))
    spatial_share = pd.to_numeric(record.get("point_label_share_250m"), errors="coerce")

    if relevance == "high":
        relevance_score = 3
    elif relevance == "medium":
        relevance_score = 2
    elif relevance == "low":
        relevance_score = 1
    else:
        relevance_score = 0

    if pd.isna(spatial_share):
        spatial_score = 0.0
        spatial_share_value = None
    else:
        spatial_share_value = float(spatial_share)
        spatial_score = 3.0 * spatial_share_value

    evidence_count = pd.to_numeric(record.get("evidence_sentences_count"), errors="coerce")
    if pd.isna(evidence_count) or int(evidence_count) <= 0:
        evidence_density_score = 0
    elif int(evidence_count) == 1:
        evidence_density_score = 1
    else:
        evidence_density_score = 2

    article_type = _normalize_optional_text(record.get("primary_article_type"))
    article_type_score = ARTICLE_TYPE_TRAIN_SCORE.get(article_type, 0)
    uncertainty_penalty = {"low": 0, "medium": 1, "high": 2}.get(uncertainty, 1)

    quality_score = (
        relevance_score
        + spatial_score
        + evidence_density_score
        + article_type_score
        - uncertainty_penalty
    )
    quality_bin = classify_quality_bin(float(quality_score))

    article_type_missing = article_type in {"", "none", "missing", "unknown", "other_or_unclear"}
    if (
        quality_score < 3
        or relevance == "none"
        or uncertainty == "high"
    ):
        recommended_use = "exclude"
    elif (
        quality_bin == "quality_very_high"
        and spatial_share_value is not None
        and spatial_share_value >= 0.9
        and relevance == "high"
    ):
        recommended_use = "use_for_evaluation_only"
    elif (
        quality_bin in {"quality_high", "quality_very_high"}
        and spatial_share_value is not None
        and spatial_share_value >= 0.8
        and relevance in {"medium", "high"}
    ):
        recommended_use = "use_for_training"
    elif (
        quality_bin == "quality_medium"
        or article_type_missing
        or uncertainty == "medium"
    ):
        recommended_use = "inspect_manually"
    else:
        recommended_use = "inspect_manually"

    return {
        "relevance_score": relevance_score,
        "spatial_score": spatial_score,
        "evidence_density_score": evidence_density_score,
        "article_type_score": article_type_score,
        "uncertainty_penalty": uncertainty_penalty,
        "quality_score": float(quality_score),
        "quality_bin": quality_bin,
        "recommended_use": recommended_use,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-experiment-dir",
        action="append",
        type=Path,
        default=None,
        help="Frozen experiment directory (repeatable).",
    )
    parser.add_argument(
        "--wiki-articles-path",
        type=Path,
        default=DEFAULT_WIKI_ARTICLES_PATH,
    )
    parser.add_argument(
        "--evidence-metadata-path",
        type=Path,
        default=DEFAULT_EVIDENCE_METADATA_PATH,
    )
    parser.add_argument(
        "--article-type-metadata-path",
        type=Path,
        default=DEFAULT_ARTICLE_TYPE_METADATA_PATH,
    )
    parser.add_argument(
        "--spatial-confidence-path",
        type=Path,
        default=DEFAULT_SPATIAL_CONFIDENCE_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", type=str, default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args(argv)


def load_wiki_articles(path: Path) -> pd.DataFrame:
    raw = read_json_file(path)
    records: list[dict[str, Any]] = []
    iterable: list[dict[str, Any]] | dict[str, Any]
    if isinstance(raw, dict):
        iterable = list(raw.values())
    elif isinstance(raw, list):
        iterable = raw
    else:
        return pd.DataFrame()

    for payload in iterable:
        if not isinstance(payload, dict):
            continue
        records.append(
            {
                "pageid": str(payload.get("pageid", "")),
                "title": payload.get("title", ""),
                "lat": payload.get("lat"),
                "lon": payload.get("lon"),
            }
        )
    if not records:
        return pd.DataFrame(columns=["pageid", "title", "lat", "lon"])
    return pd.DataFrame(records)


def _quality_mask_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    relevance = frame["landcover_relevance"].astype("string").str.lower().fillna("")
    spatial_share = pd.to_numeric(frame["point_label_share_250m"], errors="coerce")
    quality_bin = frame["quality_bin"]
    return {
        "all": pd.Series(True, index=frame.index),
        "quality_low": quality_bin == "quality_low",
        "quality_medium": quality_bin == "quality_medium",
        "quality_high": quality_bin == "quality_high",
        "quality_very_high": quality_bin == "quality_very_high",
        "quality_high_or_very_high": quality_bin.isin(["quality_high", "quality_very_high"]),
        "relevance_medium_high": relevance.isin(["medium", "high"]),
        "spatial_250m_ge_0.8": spatial_share >= 0.8,
        "relevance_medium_high_and_spatial_250m_ge_0.8": relevance.isin(["medium", "high"])
        & (spatial_share >= 0.8),
        "quality_high_or_very_high_and_spatial_250m_ge_0.8": quality_bin.isin(
            ["quality_high", "quality_very_high"]
        )
        & (spatial_share >= 0.8),
        "recommended_use_training": frame["recommended_use"] == "use_for_training",
        "recommended_use_evaluation_only": frame["recommended_use"] == "use_for_evaluation_only",
        "recommended_use_exclude": frame["recommended_use"] == "exclude",
        "article_type_high_prior": frame["primary_article_type"].astype("string").isin(
            ARTICLE_TYPE_HIGH_PRIOR
        ),
    }


def _row_subset_metric(
    subset: pd.DataFrame,
    task: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = label_universe(subset, task=task, columns=("target",))
    if task == "corine_level2":
        metrics, per_class = compute_single_label_subset_metrics(
            subset,
            labels,
            include_records_without_target=False,
            include_missing_predictions=False,
        )
    else:
        metrics = compute_multilabel_subset_metrics(
            subset,
            labels,
            require_list_targets=True,
            denominator_by_predicted=False,
        )
        per_class = []
    return {
        "n": metrics["n"],
        "n_predicted_ok": metrics["n_predicted_ok"],
        "n_parse_error": metrics["n_parse_error"],
        "coverage": metrics["coverage"],
        "accuracy": metrics.get("accuracy", ""),
        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
        "macro_precision": metrics.get("macro_precision", ""),
        "macro_recall": metrics.get("macro_recall", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "micro_precision": metrics.get("micro_precision", ""),
        "micro_recall": metrics.get("micro_recall", ""),
        "micro_f1": metrics.get("micro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "jaccard": metrics.get("jaccard", ""),
        "hamming_loss": metrics.get("hamming_loss", ""),
        "majority_accuracy": metrics.get("majority_accuracy", ""),
        "majority_balanced_accuracy": metrics.get("majority_balanced_accuracy", ""),
        "majority_macro_f1": metrics.get("majority_macro_f1", ""),
        "majority_labelset_exact_match_accuracy": metrics.get("majority_labelset_exact_match_accuracy", ""),
        "empty_set_exact_match_accuracy": metrics.get("empty_set_exact_match_accuracy", ""),
        "exact_match_accuracy": metrics.get("exact_match_accuracy", ""),
    }, [
        {**item, "label": str(item["label"])} for item in per_class
    ]


def _compute_overview_rows(
    merged: pd.DataFrame,
    mask_map: dict[str, pd.Series],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overview_rows: list[dict[str, Any]] = []
    majority_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for (model, task, text_source), rows in merged.groupby(["model", "task", "text_source"]):
        for subset_name, mask in mask_map.items():
            subset_rows = rows.loc[mask.loc[rows.index]]
            metrics, per_class = _row_subset_metric(subset_rows, task=task)
            overview_rows.append(
                {
                    "subset": subset_name,
                    "model": model,
                    "task": task,
                    "text_source": text_source,
                    **metrics,
                }
            )
            majority_rows.append(
                {
                    "model": model,
                    "task": task,
                    "text_source": text_source,
                    "subset": subset_name,
                    "n": metrics["n"],
                    "n_predicted_ok": metrics["n_predicted_ok"],
                    "n_parse_error": metrics["n_parse_error"],
                    "coverage": metrics["coverage"],
                    "majority_accuracy": metrics["majority_accuracy"],
                    "majority_balanced_accuracy": metrics["majority_balanced_accuracy"],
                    "majority_macro_f1": metrics["majority_macro_f1"],
                    "majority_labelset_exact_match_accuracy": metrics["majority_labelset_exact_match_accuracy"],
                    "empty_set_exact_match_accuracy": metrics["empty_set_exact_match_accuracy"],
                }
            )
            if task == "corine_level2":
                for item in per_class:
                    item.update(
                        {
                            "subset": subset_name,
                            "model": model,
                            "task": task,
                            "text_source": text_source,
                        }
                    )
                    per_class_rows.append(item)
    return overview_rows, majority_rows, per_class_rows


def _compute_shuffled_delta_rows(overview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["model"], row["task"], row["text_source"], row["subset"]): row
        for row in overview_rows
    }
    shuffled_pairs = shuffled_text_source_pairs({str(row["text_source"]) for row in overview_rows})
    output: list[dict[str, Any]] = []
    for row in overview_rows:
        source = row["text_source"]
        shuffled = shuffled_pairs.get(source)
        if not shuffled:
            continue
        reference = lookup.get((row["model"], row["task"], shuffled, row["subset"]))
        if not reference:
            continue
        metric = primary_metric_name(str(row["task"]), osm_metric="exact_match_accuracy")
        aligned_score = row.get(metric)
        shuffled_score = reference.get(metric)
        delta: float | str = ""
        if isinstance(aligned_score, (int, float)) and isinstance(shuffled_score, (int, float)):
            delta = float(aligned_score) - float(shuffled_score)
        output.append(
            {
                "model": row["model"],
                "task": row["task"],
                "subset": row["subset"],
                "text_source": source,
                "shuffled_text_source": shuffled,
                "primary_metric": metric,
                "aligned_score": aligned_score,
                "shuffled_score": shuffled_score,
                "delta": delta,
                "n_aligned": row.get("n", ""),
                "n_shuffled": reference.get("n", ""),
            }
        )
    return output


def _compute_model_comparison_rows(overview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in overview_rows:
        by_bucket.setdefault((row["task"], row["text_source"], row["subset"]), {})[
            row["model"]
        ] = row

    output: list[dict[str, Any]] = []
    for (task, text_source, subset), model_rows in by_bucket.items():
        models = sorted(model_rows.keys())
        if len(models) < 2:
            continue
        model_a, model_b = models[:2]
        row_a = model_rows[model_a]
        row_b = model_rows[model_b]
        metric = primary_metric_name(task, osm_metric="exact_match_accuracy")
        score_a = row_a.get(metric)
        score_b = row_b.get(metric)
        if not isinstance(score_a, (int, float)) or not isinstance(score_b, (int, float)):
            continue
        output.append(
            {
                "task": task,
                "text_source": text_source,
                "subset": subset,
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "score_a": score_a,
                "score_b": score_b,
                "delta_a_minus_b": float(score_a) - float(score_b),
                "n_a": row_a.get("n", ""),
                "n_b": row_b.get("n", ""),
            }
        )
    return output


def build_quality_rows(
    predictions: pd.DataFrame,
    wiki: pd.DataFrame,
    evidence_metadata: pd.DataFrame,
    spatial_confidence: pd.DataFrame,
    article_types: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[str, list[str]]]:
    article_types_with_flags = article_types.copy()
    evidence_metadata_with_flags = evidence_metadata.copy()
    spatial_confidence_with_flags = spatial_confidence.copy()

    if "_has_article_type_metadata" not in article_types_with_flags.columns:
        article_types_with_flags["_has_article_type_metadata"] = True
    if "_has_evidence_metadata" not in evidence_metadata_with_flags.columns:
        evidence_metadata_with_flags["_has_evidence_metadata"] = True
    if "_has_spatial_metadata" not in spatial_confidence_with_flags.columns:
        spatial_confidence_with_flags["_has_spatial_metadata"] = True

    article_types_with_flags = article_types_with_flags.rename(
        columns={"title": "_article_type_title"}
    )

    predicted_pageids = pd.DataFrame({"pageid": sorted(predictions["pageid"].astype(str).unique())})
    artifact = (
        predicted_pageids
        .merge(wiki, on="pageid", how="left")
        .merge(evidence_metadata_with_flags, on="pageid", how="left")
        .merge(spatial_confidence_with_flags, on="pageid", how="left")
        .merge(article_types_with_flags, on="pageid", how="left")
    )

    artifact["title"] = artifact["title"].fillna(artifact["_article_type_title"])

    artifact["evidence_types"] = artifact["evidence_types"].apply(normalize_string_list)
    artifact["candidate_article_types"] = artifact["candidate_article_types"].apply(
        lambda value: normalize_string_list(value)
    )
    artifact["landcover_relevance"] = artifact["landcover_relevance"].astype("string")
    artifact["uncertainty"] = artifact["uncertainty"].astype("string")
    artifact["primary_article_type"] = artifact["primary_article_type"].fillna("other_or_unclear")
    artifact["candidate_article_types"] = artifact["candidate_article_types"].apply(
        lambda values: values if values else ["other_or_unclear"]
    )
    artifact["evidence_sentences_count"] = pd.to_numeric(
        artifact["evidence_sentences_count"], errors="coerce"
    ).fillna(0).astype(int)
    artifact["landuse_evidence_summary_char_count"] = pd.to_numeric(
        artifact["landuse_evidence_summary_char_count"], errors="coerce"
    ).fillna(0).astype(int)

    quality_scores: list[dict[str, Any]] = []
    for _, row in artifact.iterrows():
        quality_scores.append(compute_quality_row(row.to_dict()))
    quality_df = pd.DataFrame(quality_scores)
    artifact = pd.concat([artifact.reset_index(drop=True), quality_df.reset_index(drop=True)], axis=1)

    osm_targets = predictions[predictions["task"] == "osm"]
    osm_by_pageid = (
        osm_targets.groupby("pageid")["target"]
        .apply(lambda values: sorted({str(value) for row in values if isinstance(row, list) for value in row}))
        .to_dict()
        if not osm_targets.empty
        else {}
    )
    artifact["osm_labels"] = artifact["pageid"].map(osm_by_pageid).apply(
        lambda values: values if isinstance(values, list) else []
    )
    artifact["corine_label"] = artifact["point_label"].fillna("")

    quality_counts = {
        "quality_low": 0,
        "quality_medium": 0,
        "quality_high": 0,
        "quality_very_high": 0,
    }
    for bin_name, count in artifact["quality_bin"].value_counts().items():
        quality_counts[str(bin_name)] = int(count)

    has_evidence_metadata = artifact["_has_evidence_metadata"].astype("boolean").fillna(False)
    has_spatial_metadata = artifact["_has_spatial_metadata"].astype("boolean").fillna(False)
    has_article_type_metadata = (
        artifact["_has_article_type_metadata"].astype("boolean").fillna(False)
    )
    missing_metadata_counts = {
        "evidence_metadata_missing": int((~has_evidence_metadata).sum()),
        "spatial_metadata_missing": int((~has_spatial_metadata).sum()),
        "article_type_metadata_missing": int((~has_article_type_metadata).sum()),
    }

    artifact = artifact.drop(
        columns=[
            "_has_evidence_metadata",
            "_has_spatial_metadata",
            "_has_article_type_metadata",
            "_article_type_title",
        ]
    )
    return artifact, quality_counts, missing_metadata_counts, osm_by_pageid


def _to_float_or_zero(value: Any) -> float:
    if pd.isna(value) or value is None:
        return 0.0
    return float(value)


def _quality_output_rows(
    quality_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for _, row in quality_rows.iterrows():
        outputs.append(
            {
                "pageid": str(row["pageid"]),
                "title": row.get("title", ""),
                "lat": row.get("lat", ""),
                "lon": row.get("lon", ""),
                "corine_label": row.get("corine_label", ""),
                "osm_labels": row.get("osm_labels", []),
                "primary_article_type": row.get("primary_article_type", "other_or_unclear"),
                "candidate_article_types": row.get("candidate_article_types", ["other_or_unclear"]),
                "landcover_relevance": row.get("landcover_relevance", ""),
                "uncertainty": row.get("uncertainty", ""),
                "evidence_types": row.get("evidence_types", []),
                "evidence_sentences_count": int(row.get("evidence_sentences_count", 0) or 0),
                "landuse_evidence_summary_char_count": int(
                    row.get("landuse_evidence_summary_char_count", 0) or 0
                ),
                "point_label_share_250m": _to_float_or_zero(row.get("point_label_share_250m")),
                "point_label_share_500m": _to_float_or_zero(row.get("point_label_share_500m")),
                "dominant_matches_point_label_250m": row.get("dominant_matches_point_label_250m", ""),
                "relevance_score": row["relevance_score"],
                "spatial_score": row["spatial_score"],
                "evidence_density_score": row["evidence_density_score"],
                "article_type_score": row["article_type_score"],
                "uncertainty_penalty": row["uncertainty_penalty"],
                "quality_score": row["quality_score"],
                "quality_bin": row["quality_bin"],
                "recommended_use": row["recommended_use"],
            }
        )
    return outputs


def _build_summary(
    quality_count: int,
    distribution: list[dict[str, Any]],
    recommended_use_distribution: list[dict[str, Any]],
    shuffled_deltas: list[dict[str, Any]],
) -> str:
    lines = [
        "# Article text supervision quality score",
        "",
        "- Analysis-only evaluation reuses frozen predictions and computes per-article supervision quality",
        "  scores before recomputing subset metrics.",
        "- Exclude precedence is conservative and can override otherwise strong rows.",
        "",
        f"- Number of quality-scored articles: {quality_count}",
        "",
        "## Quality Distribution",
    ]
    for row in distribution:
        if row["dimension"] == "quality_bin":
            lines.append(f"- {row['value']}: {row['count']}")
    lines.extend(
        [
            "",
            "## Recommended Use",
            "Precedence (ordered):",
            *[f"- {item}" for item in RECOMMENDED_USE_PRECEDENCE],
            "",
            "## Recommended Use Distribution",
        ]
    )
    for row in recommended_use_distribution:
        if row["dimension"] == "recommended_use":
            lines.append(f"- {row['value']}: {row['count']}")

    if shuffled_deltas:
        best_delta_rows = sorted(
            [
                row
                for row in shuffled_deltas
                if isinstance(row.get("delta"), (int, float))
            ],
            key=lambda row: abs(float(row["delta"])),
            reverse=True,
        )
        if best_delta_rows:
            top = best_delta_rows[0]
            lines.extend(
                [
                    "",
                    "## Strongest Shuffled Delta",
                    f"- {top['model']} {top['task']} {top['subset']} {top['text_source']}: "
                    f"{top['primary_metric']}={top['delta']}",
                ]
            )
    return "\n".join(lines) + "\n"


def evaluate(
    parent_dirs: list[Path],
    wiki_articles_path: Path,
    evidence_metadata_path: Path,
    article_type_metadata_path: Path,
    spatial_confidence_path: Path,
    output_dir: Path,
    *,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    wiki = load_wiki_articles(wiki_articles_path)
    evidence = load_evidence_metadata(evidence_metadata_path)
    spatial = load_spatial_confidence(spatial_confidence_path)
    article_types = load_article_type_metadata(article_type_metadata_path)

    prediction_frames: list[pd.DataFrame] = []
    for parent_dir in parent_dirs:
        records = load_prediction_records(
            parent_dir,
            text_sources=set(TEXT_SOURCES),
            normalize_targets=True,
        )
        if records.empty:
            continue
        records["model"] = infer_model_for_records(
            records,
            parent_dir,
            metadata_keys=("model", "model_repo_id"),
        )
        records["source_parent_experiment_dir"] = str(parent_dir)
        prediction_frames.append(records)

    if not prediction_frames:
        raise ValueError("No prediction records loaded from input experiments.")
    predictions = pd.concat(prediction_frames, ignore_index=True)

    quality_articles, quality_counts, missing_metadata_counts, _ = build_quality_rows(
        predictions,
        wiki=wiki,
        evidence_metadata=evidence,
        spatial_confidence=spatial,
        article_types=article_types,
    )
    for spatial_column in ["point_label_share_250m", "point_label_share_500m", "dominant_matches_point_label_250m"]:
        if spatial_column not in quality_articles.columns:
            quality_articles[spatial_column] = pd.NA
    quality_rows = _quality_output_rows(quality_articles)

    merged = predictions.merge(
        quality_articles[
            [
                "pageid",
                "relevance_score",
                "spatial_score",
                "evidence_density_score",
                "article_type_score",
                "uncertainty_penalty",
                "quality_score",
                "quality_bin",
                "recommended_use",
                "landcover_relevance",
                "uncertainty",
                "evidence_types",
                "evidence_sentences_count",
                "landuse_evidence_summary_char_count",
                "point_label_share_250m",
                "point_label_share_500m",
                "dominant_matches_point_label_250m",
                "primary_article_type",
                "candidate_article_types",
                "corine_label",
                "osm_labels",
            ]
        ],
        on="pageid",
        how="left",
    )

    mask_map = _quality_mask_map(merged)

    overview_rows, majority_rows, per_class_rows = _compute_overview_rows(merged, mask_map)
    overview_rows = sorted(overview_rows, key=lambda row: (row["model"], row["task"], row["text_source"], row["subset"]))
    majority_rows = sorted(
        majority_rows,
        key=lambda row: (row["model"], row["task"], row["text_source"], row["subset"]),
    )
    per_class_rows = sorted(
        per_class_rows,
        key=lambda row: (row["model"], row["task"], row["text_source"], row["subset"], str(row.get("label"))),
    )

    shuffled_delta_rows = _compute_shuffled_delta_rows(overview_rows)
    model_comparison_rows = _compute_model_comparison_rows(overview_rows)
    simple_filter_rows = [
        row
        for row in overview_rows
        if row["subset"]
        in {
            "relevance_medium_high",
            "spatial_250m_ge_0.8",
            "relevance_medium_high_and_spatial_250m_ge_0.8",
            "article_type_high_prior",
            "quality_high_or_very_high",
            "quality_high_or_very_high_and_spatial_250m_ge_0.8",
        }
    ]

    quality_bin_distribution = [
        {"dimension": "quality_bin", "value": bin_name, "count": int(count)}
        for bin_name, count in quality_articles["quality_bin"].value_counts().items()
    ]
    recommended_use_distribution = [
        {"dimension": "recommended_use", "value": value, "count": int(count)}
        for value, count in quality_articles["recommended_use"].value_counts().items()
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_dict_rows_csv_atomic(
        output_dir / "quality_scores.csv",
        quality_rows,
        columns=QUALITY_SCORE_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "quality_scores.md",
        rows=quality_rows,
        title="Quality Scores",
        columns=QUALITY_SCORE_COLUMNS,
    )
    write_dict_rows_csv_atomic(
        output_dir / "candidate_training_pairs_by_quality.csv",
        quality_rows,
        columns=QUALITY_SCORE_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "candidate_training_pairs_by_quality.md",
        rows=quality_rows,
        title="Candidate Training Pairs by Quality",
        columns=QUALITY_SCORE_COLUMNS,
    )
    write_dict_rows_csv_atomic(
        output_dir / "quality_bin_distribution.csv",
        quality_bin_distribution,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "quality_bin_distribution.md",
        rows=quality_bin_distribution,
        title="Quality Bin Distribution",
    )
    write_dict_rows_csv_atomic(
        output_dir / "recommended_use_distribution.csv",
        recommended_use_distribution,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "recommended_use_distribution.md",
        rows=recommended_use_distribution,
        title="Recommended Use Distribution",
    )
    write_dict_rows_csv_atomic(
        output_dir / "overview_by_quality_bin.csv",
        overview_rows,
        columns=OUTPUT_METRIC_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "overview_by_quality_bin.md",
        rows=overview_rows,
        title="Overview by Quality Bin",
        columns=OUTPUT_METRIC_COLUMNS,
    )
    # Keep a compact subset for readability in the parallel file.
    write_dict_rows_csv_atomic(
        output_dir / "overview_by_quality_and_spatial.csv",
        [
            row
            for row in overview_rows
            if str(row["subset"]).endswith("and_spatial_250m_ge_0.8")
            or str(row["subset"]) in {"all", "relevance_medium_high", "quality_medium", "quality_high", "quality_very_high"}
        ],
        columns=OUTPUT_METRIC_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "overview_by_quality_and_spatial.md",
        rows=[
            row
            for row in overview_rows
            if str(row["subset"]).endswith("and_spatial_250m_ge_0.8")
            or str(row["subset"]) in {"all", "relevance_medium_high", "quality_medium", "quality_high", "quality_very_high"}
        ],
        title="Overview by Quality and Spatial Subset",
        columns=OUTPUT_METRIC_COLUMNS,
    )
    write_dict_rows_csv_atomic(
        output_dir / "overview_comparison_simple_filters.csv",
        simple_filter_rows,
        columns=OUTPUT_METRIC_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "overview_comparison_simple_filters.md",
        rows=simple_filter_rows,
        title="Overview by Simple Filter Comparison",
        columns=OUTPUT_METRIC_COLUMNS,
    )
    write_dict_rows_csv_atomic(
        output_dir / "shuffled_delta_by_quality_bin.csv",
        shuffled_delta_rows,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "shuffled_delta_by_quality_bin.md",
        rows=shuffled_delta_rows,
        title="Shuffled Deltas by Quality Bin",
    )
    write_dict_rows_csv_atomic(
        output_dir / "majority_baselines_by_quality_bin.csv",
        majority_rows,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "majority_baselines_by_quality_bin.md",
        rows=majority_rows,
        title="Majority Baselines by Quality Bin",
    )
    write_dict_rows_csv_atomic(
        output_dir / "per_class_corine_by_quality_bin.csv",
        [row for row in per_class_rows if row["task"] == "corine_level2"],
    )
    write_dict_rows_markdown_atomic(
        output_dir / "per_class_corine_by_quality_bin.md",
        rows=[row for row in per_class_rows if row["task"] == "corine_level2"],
        title="Per-class CORINE by Quality Bin",
    )
    write_dict_rows_csv_atomic(
        output_dir / "model_comparison_by_quality_bin.csv",
        model_comparison_rows,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "model_comparison_by_quality_bin.md",
        rows=model_comparison_rows,
        title="Model Comparison by Quality Bin",
    )

    write_text_atomic(
        output_dir / "summary.md",
        _build_summary(
            quality_count=len(quality_articles),
            distribution=quality_bin_distribution,
            recommended_use_distribution=recommended_use_distribution,
            shuffled_deltas=shuffled_delta_rows,
        ),
    )

    write_json_atomic(
        output_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "no_llm_rerun": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_paths": {
                "wiki_articles_path": str(wiki_articles_path),
                "evidence_metadata_path": str(evidence_metadata_path),
                "article_type_metadata_path": str(article_type_metadata_path),
                "spatial_confidence_path": str(spatial_confidence_path),
            },
            "models": sorted(set(predictions["model"])),
            "tasks_included": sorted(set(predictions["task"])),
            "text_sources_included": sorted(set(predictions["text_source"])),
            "parent_experiment_dirs": sorted({str(path) for path in parent_dirs}),
            "number_of_articles_scored": int(len(quality_articles)),
            "scoring": {
                "relevance_score": {"none": 0, "low": 1, "medium": 2, "high": 3, "missing_or_unknown": 0},
                "spatial_score": "3 * point_label_share_250m",
                "evidence_density_score": {
                    "missing_or_0": 0,
                    "1": 1,
                    ">=2": 2,
                },
                "article_type_score": {
                    "agriculture_or_vineyard": 2,
                    "natural_landscape": 2,
                    "water_feature": 2,
                    "settlement_or_administrative": 1,
                    "built_or_cultural_site": 1,
                    "other_or_unclear": 0,
                    "missing/unknown": 0,
                },
                "uncertainty_penalty": {
                    "low": 0,
                    "medium": 1,
                    "high": 2,
                    "missing_or_unknown": 1,
                },
                "quality_formula": QUALITY_SCORE_FORMULA,
            },
            "bin_definitions": {
                "quality_low": "score < 3",
                "quality_medium": "3 <= score < 5",
                "quality_high": "5 <= score < 7",
                "quality_very_high": "score >= 7",
            },
            "recommended_use_precedence": RECOMMENDED_USE_PRECEDENCE,
            "metrics_computed": [
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "weighted_f1",
                "jaccard",
                "hamming_loss",
                "majority_accuracy",
                "majority_balanced_accuracy",
                "majority_macro_f1",
                "majority_labelset_exact_match_accuracy",
                "empty_set_exact_match_accuracy",
                "exact_match_accuracy",
            ],
            "quality_bin_counts": quality_counts,
            "missing_metadata_counts": missing_metadata_counts,
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        parent_dirs=args.parent_experiment_dir or DEFAULT_PARENT_DIRS,
        wiki_articles_path=args.wiki_articles_path,
        evidence_metadata_path=args.evidence_metadata_path,
        article_type_metadata_path=args.article_type_metadata_path,
        spatial_confidence_path=args.spatial_confidence_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
