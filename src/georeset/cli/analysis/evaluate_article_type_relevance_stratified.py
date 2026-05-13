"""Evaluate existing frozen predictions stratified by article type and relevance/spatial."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.analysis.article_type_classifier import ARTICLE_TYPE_PREFERENCE
from georeset.analysis.article_type_metadata_loading import load_article_type_metadata
from georeset.analysis.evaluation_metrics import (
    compute_multilabel_subset_metrics,
    compute_single_label_subset_metrics,
)
from georeset.analysis.evidence_metadata_loading import load_evidence_metadata
from georeset.analysis.prediction_loading import infer_model_from_metadata, load_prediction_records
from georeset.analysis.spatial_confidence_loading import load_spatial_confidence
from georeset.classification.labels import CORINE_LEVEL2_DESCRIPTIONS
from georeset.utils.json_io import (
    write_dict_rows_csv_atomic,
    write_dict_rows_markdown_atomic,
    write_json_atomic,
    write_text_atomic,
)

DEFAULT_PARENT_DIRS = [
    Path("data/experiments/article_text_classification_e2e_with_shuffled_control_v1"),
    Path(
        "data/experiments/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
    ),
]
DEFAULT_ARTICLE_TYPE_METADATA_PATH = Path(
    "data/experiments/article_text_classification_article_type_relevance_stratified_v1/article_type_metadata.json"
)
DEFAULT_EVIDENCE_METADATA_PATH = Path("data/wiki/article_landuse_evidence_summaries.json")
DEFAULT_SPATIAL_CONFIDENCE_PATH = Path(
    "data/experiments/corine_spatial_confidence_v1/spatial_confidence.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/experiments/article_text_classification_article_type_relevance_stratified_v1")
DEFAULT_EXPERIMENT_ID = "article_text_classification_article_type_relevance_stratified_v1"

TEXT_SOURCES = [
    "summary",
    "summary_no_place",
    "content",
    "summary_shuffled",
    "summary_no_place_shuffled",
    "content_shuffled",
]
SHUFFLED_TEXT_SOURCE_PAIRS = {
    "summary": "summary_shuffled",
    "summary_no_place": "summary_no_place_shuffled",
    "content": "content_shuffled",
}
SPATIAL_THRESHOLD = "point_label_share_250m_ge_0.8"
RELEVANCE_SUBSET_LABELS = [
    "all",
    "relevance_none",
    "relevance_low",
    "relevance_medium",
    "relevance_high",
    "relevance_low_medium_high",
    "relevance_medium_high",
]
ARTICLE_TYPE_ASSIGNMENT_COLUMNS = [
    "pageid",
    "title",
    "primary_article_type",
    "candidate_article_types",
    "matched_categories",
    "matched_rules",
    "all_categories_count",
    "has_categories",
]


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
        "--article-type-metadata-path",
        type=Path,
        default=DEFAULT_ARTICLE_TYPE_METADATA_PATH,
    )
    parser.add_argument(
        "--evidence-metadata-path",
        type=Path,
        default=DEFAULT_EVIDENCE_METADATA_PATH,
    )
    parser.add_argument(
        "--spatial-confidence-path",
        type=Path,
        default=DEFAULT_SPATIAL_CONFIDENCE_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", type=str, default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args(argv)


def _metric_name(task: str) -> str:
    return "balanced_accuracy" if task == "corine_level2" else "exact_match_accuracy"


def define_relevance_subsets(records: pd.DataFrame) -> dict[str, pd.Series]:
    relevance = records.get("landcover_relevance", pd.Series([""] * len(records), index=records.index)).astype(
        "string"
    ).str.lower()
    return {
        "all": pd.Series(True, index=records.index),
        "relevance_none": relevance == "none",
        "relevance_low": relevance == "low",
        "relevance_medium": relevance == "medium",
        "relevance_high": relevance == "high",
        "relevance_low_medium_high": relevance.isin(["low", "medium", "high"]),
        "relevance_medium_high": relevance.isin(["medium", "high"]),
    }


def define_spatial_subsets(records: pd.DataFrame) -> dict[str, pd.Series]:
    if "point_label_share_250m" not in records.columns:
        return {
            SPATIAL_THRESHOLD: pd.Series(False, index=records.index),
            "all": pd.Series(True, index=records.index),
        }
    spatial = pd.to_numeric(records["point_label_share_250m"], errors="coerce")
    return {
        "all": pd.Series(True, index=records.index),
        SPATIAL_THRESHOLD: spatial >= 0.8,
    }


def define_article_type_subsets(records: pd.DataFrame) -> dict[str, pd.Series]:
    if "primary_article_type" not in records.columns:
        return {}
    article_types = pd.Series(records["primary_article_type"], index=records.index).fillna("other_or_unclear")
    subsets: dict[str, pd.Series] = {}
    for article_type in ARTICLE_TYPE_PREFERENCE:
        subsets[f"article_type:{article_type}"] = article_types.astype(str) == article_type
    return subsets


def _label_universe(records: pd.DataFrame, task: str) -> list[str]:
    labels: set[str] = set()
    for values in records["target"]:
        if isinstance(values, list):
            for value in values:
                labels.add(str(value))
        elif values is not None:
            labels.add(str(values))
    if task == "corine_level2":
        return sorted(CORINE_LEVEL2_DESCRIPTIONS)
    return sorted(labels)


def _metric_rows_by_subset(
    records: pd.DataFrame,
    subset_name: str,
    task: str,
    labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if task == "corine_level2":
        metrics, per_class = compute_single_label_subset_metrics(
            records,
            labels,
            include_records_without_target=False,
            include_missing_predictions=False,
        )
    else:
        metrics = compute_multilabel_subset_metrics(
            records,
            labels,
            require_list_targets=True,
            denominator_by_predicted=False,
        )
        per_class = []
    row = {
        "subset": subset_name,
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
    }
    per_class_rows: list[dict[str, Any]] = []
    for item in per_class:
        per_class_rows.append({**{"subset": subset_name}, **item})
    return row, per_class_rows


def _subset_by_mask(records: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return records.loc[mask.reindex(records.index, fill_value=False)]


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("task", "")),
        str(row.get("text_source", "")),
        str(row.get("subset", "")),
    )


def _build_delta_rows(
    overview_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["model"], row["task"], row["text_source"], row["subset"]): row for row in overview_rows
    }
    output: list[dict[str, Any]] = []
    for row in overview_rows:
        model = row["model"]
        task = row["task"]
        text_source = str(row["text_source"])
        shuffled = SHUFFLED_TEXT_SOURCE_PAIRS.get(text_source)
        if not shuffled:
            continue
        ref = lookup.get((model, task, shuffled, row["subset"]))
        if not ref:
            continue
        metric = _metric_name(task)
        metric_aligned = row.get(metric)
        metric_shuffled = ref.get(metric)
        delta: float | str = ""
        if isinstance(metric_aligned, (int, float)) and isinstance(metric_shuffled, (int, float)):
            delta = float(metric_aligned) - float(metric_shuffled)
        output.append(
            {
                "model": model,
                "task": task,
                "text_source": text_source,
                "shuffled_text_source": shuffled,
                "subset": row["subset"],
                "primary_metric": metric,
                "aligned_score": metric_aligned,
                "shuffled_score": metric_shuffled,
                "delta": delta,
                "n_aligned": row.get("n", ""),
                "n_shuffled": ref.get("n", ""),
            }
        )
    return output


def _build_article_type_assignment_outputs(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return [
        {
            "pageid": row["pageid"],
            "title": row.get("title", ""),
            "primary_article_type": row.get("primary_article_type", "other_or_unclear"),
            "candidate_article_types": row.get("candidate_article_types", []),
            "matched_categories": row.get("matched_categories", []),
            "matched_rules": row.get("matched_rules", []),
            "all_categories_count": row.get("all_categories_count", 0),
            "has_categories": row.get("has_categories", False),
        }
        for _, row in df[[c for c in ARTICLE_TYPE_ASSIGNMENT_COLUMNS if c in df.columns]].iterrows()
    ]


def _build_audit_sample(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    sorted_df = df.sort_values(["primary_article_type", "pageid", "title"])
    rows: list[dict[str, Any]] = []
    for _article_type, group in sorted_df.groupby("primary_article_type"):
        sample = group.head(20)
        for _, row in sample.iterrows():
            rows.append(
                {
                    "pageid": row["pageid"],
                    "title": row.get("title", ""),
                    "primary_article_type": row.get("primary_article_type", "other_or_unclear"),
                    "candidate_article_types": row.get("candidate_article_types", []),
                    "matched_categories": row.get("matched_categories", []),
                    "matched_rules": row.get("matched_rules", []),
                    "all_categories_count": row.get("all_categories_count", 0),
                    "has_categories": row.get("has_categories", False),
                }
            )
    return rows


def _build_manifest_join_counts(
    merged: pd.DataFrame,
    *,
    prediction_records: pd.DataFrame,
    article_type_metadata: pd.DataFrame,
    evidence_loaded: bool,
    spatial_loaded: bool,
    article_type_loaded: bool,
) -> dict[str, int]:
    n_prediction_records_loaded = int(len(prediction_records))
    n_unique_prediction_pageids = int(pd.Series(prediction_records["pageid"]).astype(str).nunique())
    n_article_type_metadata_records = int(len(article_type_metadata))
    n_unique_article_type_pageids = int(
        pd.Series(article_type_metadata["pageid"]).astype(str).nunique()
    ) if article_type_loaded else 0

    if evidence_loaded:
        n_predictions_missing_evidence_metadata = int(pd.Series(merged["landcover_relevance"]).isna().sum())
    else:
        n_predictions_missing_evidence_metadata = n_prediction_records_loaded

    if spatial_loaded:
        n_predictions_missing_spatial_confidence = int(
            pd.to_numeric(merged["point_label_share_250m"], errors="coerce").isna().sum()
        )
    else:
        n_predictions_missing_spatial_confidence = n_prediction_records_loaded

    if article_type_loaded:
        n_predictions_missing_article_type_metadata = int(merged["primary_article_type"].isna().sum())
    else:
        n_predictions_missing_article_type_metadata = n_prediction_records_loaded

    return {
        "n_prediction_records_loaded": n_prediction_records_loaded,
        "n_unique_prediction_pageids": n_unique_prediction_pageids,
        "n_article_type_metadata_records": n_article_type_metadata_records,
        "n_unique_article_type_pageids": n_unique_article_type_pageids,
        "n_predictions_missing_article_type_metadata": n_predictions_missing_article_type_metadata,
        "n_predictions_missing_evidence_metadata": n_predictions_missing_evidence_metadata,
        "n_predictions_missing_spatial_confidence": n_predictions_missing_spatial_confidence,
    }


def evaluate(
    parent_dirs: list[Path],
    article_type_metadata_path: Path,
    evidence_metadata_path: Path,
    spatial_confidence_path: Path,
    output_dir: Path,
    *,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    evidence = load_evidence_metadata(evidence_metadata_path)
    spatial = load_spatial_confidence(spatial_confidence_path, allow_missing_pageid=True)
    article_types = load_article_type_metadata(article_type_metadata_path)

    frames: list[pd.DataFrame] = []
    for parent_dir in parent_dirs:
        records = load_prediction_records(
            parent_dir,
            text_sources=TEXT_SOURCES,
            include_source_dir=True,
        )
        if records.empty:
            continue
        records["model"] = records["metadata"].apply(
            lambda value, parent_dir=parent_dir: infer_model_from_metadata(
                value if isinstance(value, Mapping) else {},
                parent_dir,
                metadata_keys=("model", "model_repo_id"),
            )
        )
        frames.append(records)

    if not frames:
        raise ValueError("No predictions loaded from parent experiment directories.")

    merged = pd.concat(frames, ignore_index=True)
    prediction_records = merged.copy(deep=True)
    if evidence is not None and not evidence.empty:
        merged = merged.merge(evidence, on="pageid", how="left")
    else:
        merged["landcover_relevance"] = None
        merged["uncertainty"] = None
        merged["evidence_types"] = [[] for _ in range(len(merged))]
        merged["evidence_sentences_count"] = 0

    if spatial is not None and not spatial.empty:
        merged = merged.merge(spatial, on="pageid", how="left")
    else:
        merged["point_label_share_250m"] = pd.Series([float("nan")] * len(merged))

    evidence_loaded = evidence is not None and not evidence.empty
    spatial_loaded = spatial is not None and not spatial.empty
    article_type_loaded = not article_types.empty

    if article_types.empty:
        merged["primary_article_type"] = "other_or_unclear"
        merged["candidate_article_types"] = [[] for _ in range(len(merged))]
        merged["matched_categories"] = [[] for _ in range(len(merged))]
        merged["matched_rules"] = [[] for _ in range(len(merged))]
        merged["all_categories_count"] = 0
        merged["has_categories"] = False
        article_types = pd.DataFrame()
    else:
        merged = merged.merge(article_types, on="pageid", how="left")
        merged["primary_article_type"] = merged["primary_article_type"].fillna("other_or_unclear")
        merged["candidate_article_types"] = merged["candidate_article_types"].apply(
            lambda value: value if isinstance(value, list) else ["other_or_unclear"]
        )
        merged["matched_categories"] = merged["matched_categories"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        merged["matched_rules"] = merged["matched_rules"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        merged["all_categories_count"] = pd.to_numeric(merged["all_categories_count"], errors="coerce").fillna(0).astype(int)
        merged["has_categories"] = merged["has_categories"].fillna(False).astype(bool)

    manifest_join_counts = _build_manifest_join_counts(
        merged,
        prediction_records=prediction_records,
        article_type_metadata=article_types,
        evidence_loaded=evidence_loaded,
        spatial_loaded=spatial_loaded,
        article_type_loaded=article_type_loaded,
    )

    merged["evidence_types"] = merged["evidence_types"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    merged["evidence_sentences_count"] = pd.to_numeric(
        merged["evidence_sentences_count"], errors="coerce"
    ).fillna(0).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)

    article_type_subsets = define_article_type_subsets(merged)
    relevance_subsets = define_relevance_subsets(merged)
    spatial_subsets = define_spatial_subsets(merged)

    # Restrict to required spatial label.
    if "all" not in spatial_subsets:
        spatial_subsets["all"] = pd.Series(True, index=merged.index)
    relevance_relabel = {
        f"relevance:{key.removeprefix('relevance_')}": value for key, value in relevance_subsets.items()
    }
    spatial_relabel = {
        f"spatial:{key}" if key != "all" else "spatial:all": value for key, value in spatial_subsets.items()
    }

    overview_rows: list[dict[str, Any]] = []
    majority_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for (model, task, text_source), group in merged.groupby(["model", "task", "text_source"], sort=False):
        labels = _label_universe(group, task)

        for subset_name, mask in article_type_subsets.items():
            rows = _metric_rows_by_subset(_subset_by_mask(group, mask), subset_name, task, labels)
            metrics, per_class = rows
            row = {"model": model, "task": task, "text_source": text_source}
            row.update(metrics)
            overview_rows.append(row)
            majority_rows.append({
                key: row.get(key, "")
                for key in [
                    "model",
                    "task",
                    "text_source",
                    "subset",
                    "majority_accuracy",
                    "majority_balanced_accuracy",
                    "majority_macro_f1",
                    "majority_labelset_exact_match_accuracy",
                    "empty_set_exact_match_accuracy",
                ]
            })
            per_class_rows.extend({**{"model": model, "task": task, "text_source": text_source}, **item} for item in per_class)

        for subset_key, mask in relevance_relabel.items():
            rows = _metric_rows_by_subset(_subset_by_mask(group, mask), subset_key, task, labels)
            metrics, per_class = rows
            row = {"model": model, "task": task, "text_source": text_source, "subset": subset_key}
            row.update(metrics)
            overview_rows.append(row)
            majority_rows.append({
                key: row.get(key, "")
                for key in [
                    "model",
                    "task",
                    "text_source",
                    "subset",
                    "majority_accuracy",
                    "majority_balanced_accuracy",
                    "majority_macro_f1",
                    "majority_labelset_exact_match_accuracy",
                    "empty_set_exact_match_accuracy",
                ]
            })
            per_class_rows.extend({**{"model": model, "task": task, "text_source": text_source}, **item} for item in per_class)

        for subset_key, mask in spatial_relabel.items():
            rows = _metric_rows_by_subset(_subset_by_mask(group, mask), subset_key, task, labels)
            metrics, per_class = rows
            row = {"model": model, "task": task, "text_source": text_source, "subset": subset_key}
            row.update(metrics)
            overview_rows.append(row)
            majority_rows.append({
                key: row.get(key, "")
                for key in [
                    "model",
                    "task",
                    "text_source",
                    "subset",
                    "majority_accuracy",
                    "majority_balanced_accuracy",
                    "majority_macro_f1",
                    "majority_labelset_exact_match_accuracy",
                    "empty_set_exact_match_accuracy",
                ]
                })
            per_class_rows.extend({**{"model": model, "task": task, "text_source": text_source}, **item} for item in per_class)

        for article_label, article_mask in article_type_subsets.items():
            for relevance_key, relevance_mask in relevance_relabel.items():
                subset_name = f"{article_label}|{relevance_key}"
                mask = article_mask & relevance_mask
                rows = _metric_rows_by_subset(
                    _subset_by_mask(group, mask), subset_name, task, labels
                )
                metrics, per_class = rows
                row = {"model": model, "task": task, "text_source": text_source, "subset": subset_name}
                row.update(metrics)
                overview_rows.append(row)
                majority_rows.append({
                    key: row.get(key, "")
                    for key in [
                        "model",
                        "task",
                        "text_source",
                        "subset",
                        "majority_accuracy",
                        "majority_balanced_accuracy",
                        "majority_macro_f1",
                        "majority_labelset_exact_match_accuracy",
                        "empty_set_exact_match_accuracy",
                    ]
                })
                per_class_rows.extend({**{"model": model, "task": task, "text_source": text_source}, **item} for item in per_class)

                for spatial_key, spatial_mask in spatial_relabel.items():
                    subset_name = f"{article_label}|{relevance_key}|{spatial_key}"
                    mask = article_mask & relevance_mask & spatial_mask
                    rows = _metric_rows_by_subset(
                        _subset_by_mask(group, mask), subset_name, task, labels
                    )
                    metrics, per_class = rows
                    row = {"model": model, "task": task, "text_source": text_source, "subset": subset_name}
                    row.update(metrics)
                    overview_rows.append(row)
                    majority_rows.append({
                        key: row.get(key, "")
                        for key in [
                            "model",
                            "task",
                            "text_source",
                            "subset",
                            "majority_accuracy",
                            "majority_balanced_accuracy",
                            "majority_macro_f1",
                            "majority_labelset_exact_match_accuracy",
                            "empty_set_exact_match_accuracy",
                        ]
                    })
                    per_class_rows.extend({**{"model": model, "task": task, "text_source": text_source}, **item} for item in per_class)

                for spatial_key, spatial_mask in spatial_relabel.items():
                    subset_name = f"{article_label}|{spatial_key}"
                    mask = article_mask & spatial_mask
                    rows = _metric_rows_by_subset(_subset_by_mask(group, mask), subset_name, task, labels)
                    metrics, per_class = rows
                    row = {
                        "model": model,
                        "task": task,
                        "text_source": text_source,
                        "subset": subset_name,
                    }
                    row.update(metrics)
                    overview_rows.append(row)
                    majority_rows.append({
                        key: row.get(key, "")
                        for key in [
                            "model",
                            "task",
                            "text_source",
                            "subset",
                            "majority_accuracy",
                            "majority_balanced_accuracy",
                            "majority_macro_f1",
                            "majority_labelset_exact_match_accuracy",
                            "empty_set_exact_match_accuracy",
                        ]
                    })
                    per_class_rows.extend(
                        {**{"model": model, "task": task, "text_source": text_source}, **item}
                        for item in per_class
                    )

    # Article-type specific overview and spatial+relevance family output file.
    article_type_relevance_spatial = [
        row
        for row in overview_rows
        if "article_type:" in str(row["subset"]) and "|" in str(row["subset"]) and "relevance:" in str(row["subset"]) and "spatial:" in str(row["subset"])
    ]

    assignments_output = _build_article_type_assignment_outputs(article_types if not article_types.empty else merged)
    audit_sample = _build_audit_sample(article_types if not article_types.empty else merged)

    write_dict_rows_csv_atomic(
        output_dir / "article_type_assignments.csv",
        assignments_output,
        columns=ARTICLE_TYPE_ASSIGNMENT_COLUMNS,
    )
    write_dict_rows_csv_atomic(
        output_dir / "article_type_assignment_audit_sample.csv",
        audit_sample,
        columns=ARTICLE_TYPE_ASSIGNMENT_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "article_type_assignments.md",
        title="Article Type Assignments",
        rows=assignments_output,
        columns=ARTICLE_TYPE_ASSIGNMENT_COLUMNS,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "article_type_assignment_audit_sample.md",
        title="Article Type Assignment Audit Sample",
        rows=audit_sample,
        columns=ARTICLE_TYPE_ASSIGNMENT_COLUMNS,
    )

    sorted_overview_rows = sorted(overview_rows, key=_row_sort_key)
    write_dict_rows_csv_atomic(output_dir / "overview_by_article_type.csv", sorted_overview_rows)
    write_dict_rows_csv_atomic(
        output_dir / "overview_by_article_type_relevance_spatial.csv",
        sorted(article_type_relevance_spatial, key=_row_sort_key),
    )
    write_dict_rows_csv_atomic(
        output_dir / "majority_baselines_by_article_type.csv", sorted(majority_rows, key=_row_sort_key)
    )
    write_dict_rows_csv_atomic(
        output_dir / "per_class_corine_by_article_type.csv", sorted(per_class_rows, key=_row_sort_key)
    )

    write_dict_rows_markdown_atomic(
        output_dir / "overview_by_article_type.md",
        title="Overview by Article-Type Subset",
        rows=sorted_overview_rows,
    )
    write_dict_rows_markdown_atomic(
        output_dir / "overview_by_article_type_relevance_spatial.md",
        title="Article-Type x Relevance x Spatial Overview",
        rows=sorted(article_type_relevance_spatial, key=_row_sort_key),
    )
    write_dict_rows_markdown_atomic(
        output_dir / "majority_baselines_by_article_type.md",
        title="Majority Baselines by Article-Type Subset",
        rows=sorted(majority_rows, key=_row_sort_key),
    )
    write_dict_rows_markdown_atomic(
        output_dir / "per_class_corine_by_article_type.md",
        title="Per-Class CORINE by Article-Type",
        rows=sorted(per_class_rows, key=_row_sort_key),
    )

    delta_rows = _build_delta_rows(sorted_overview_rows)
    write_dict_rows_csv_atomic(
        output_dir / "shuffled_delta_by_article_type.csv",
        sorted(delta_rows, key=_row_sort_key),
    )
    write_dict_rows_markdown_atomic(
        output_dir / "shuffled_delta_by_article_type.md",
        title="Shuffled Delta by Article-Type Subset",
        rows=sorted(delta_rows, key=_row_sort_key),
    )

    # Distribution by article type and source/task
    if article_types.empty:
        distribution_rows = []
    else:
        for (task, text_source, model), group in merged.groupby(["task", "text_source", "model"], sort=False):
            counts = group["primary_article_type"].value_counts().to_dict()
            for article_type, count in sorted(counts.items()):
                distribution_rows.append(
                    {
                        "task": task,
                        "text_source": text_source,
                        "model": model,
                        "primary_article_type": article_type,
                        "count": int(count),
                    }
                )
    write_dict_rows_csv_atomic(output_dir / "article_type_distribution.csv", distribution_rows)
    write_dict_rows_markdown_atomic(
        output_dir / "article_type_distribution.md",
        title="Article Type Distribution",
        rows=sorted(
            distribution_rows,
            key=lambda row: (
                str(row["task"]),
                str(row["text_source"]),
                str(row["model"]),
                str(row["primary_article_type"]),
            ),
            ),
    )

    write_json_atomic(
        output_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "no_llm_rerun": True,
            "input_paths": {
                "article_type_metadata_path": str(article_type_metadata_path),
                "evidence_metadata_path": str(evidence_metadata_path),
                "spatial_confidence_path": str(spatial_confidence_path),
            },
            "parent_experiment_dirs": sorted({str(path) for path in parent_dirs}),
            "article_type_taxonomy": ARTICLE_TYPE_PREFERENCE,
            **manifest_join_counts,
            "subset_families": sorted(
                {
                    "article_type",
                    "relevance",
                    "spatial",
                    "article_type+relevance",
                    "article_type+spatial",
                    "article_type+relevance+spatial",
                }
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )

    summary = [
        f"# {experiment_id}",
        "",
        "- Evaluated frozen predictions with article-type and metadata stratification.",
        "- No LLM reruns were triggered.",
    ]
    write_text_atomic(output_dir / "summary.md", "\n".join(summary) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        parent_dirs=args.parent_experiment_dir or DEFAULT_PARENT_DIRS,
        article_type_metadata_path=args.article_type_metadata_path,
        evidence_metadata_path=args.evidence_metadata_path,
        spatial_confidence_path=args.spatial_confidence_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
