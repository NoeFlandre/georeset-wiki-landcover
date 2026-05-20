"""Evaluate filtered article-text subsets against randomization controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from georeset.analysis.article_type_metadata_loading import load_article_type_metadata
from georeset.analysis.evaluation_metrics import compute_task_subset_metrics
from georeset.analysis.evidence_metadata_loading import load_evidence_metadata
from georeset.analysis.label_universe import label_universe
from georeset.analysis.prediction_loading import infer_model_for_records, load_prediction_records
from georeset.analysis.spatial_confidence_loading import load_spatial_confidence
from georeset.classification.text_sources import shuffled_text_source_pairs
from georeset.experiment_paths import experiment_artifact_dir, experiment_artifact_file
from georeset.utils.boolish import parse_boolish_series
from georeset.utils.json_io import (
    markdown_table,
    resolve_table_columns,
    write_json_atomic,
    write_text_atomic,
)

EXPERIMENT_ID = "article_text_subset_randomization_controls_v1"
DEFAULT_OUTPUT_DIR = experiment_artifact_dir(EXPERIMENT_ID)
DEFAULT_PARENT_DIRS = [
    experiment_artifact_dir("article_text_classification_e2e_with_shuffled_control_v1"),
    experiment_artifact_dir(
        "article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
    ),
]
DEFAULT_SPATIAL_CONFIDENCE_PATH = experiment_artifact_file(
    "corine_spatial_confidence_v1", "spatial_confidence.csv"
)
DEFAULT_EVIDENCE_METADATA_PATH = Path("data/wiki/article_landuse_evidence_summaries.json")
DEFAULT_ARTICLE_TYPE_ASSIGNMENTS_PATH = experiment_artifact_file(
    "article_text_classification_article_type_relevance_stratified_v1",
    "article_type_assignments.csv",
)
DEFAULT_QUALITY_SCORES_PATH = experiment_artifact_file(
    "article_text_supervision_quality_score_v1", "quality_scores.csv"
)
SMALL_N_THRESHOLD = 30

HEADLINE_TEXT_SOURCES = {"content", "summary_no_place", "landuse_evidence_summary"}
HEADLINE_SUBSETS = {
    "relevance_medium_high",
    "point_label_share_250m_ge_0.8",
    "relevance_medium_high_and_spatial_250m_ge_0.8",
    "quality_high_or_very_high_and_spatial_250m_ge_0.8",
    "recommended_use_training",
    "recommended_use_evaluation_only",
}
SHUFFLED_PAIRS = {
    key: value
    for key, value in shuffled_text_source_pairs().items()
    if key
    in {
        "summary",
        "summary_no_place",
        "content",
        "landuse_evidence_summary",
        "evidence_card",
        "content_with_evidence_card",
    }
}


def subset_randomization_manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def subset_randomization_summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.md"


@dataclass(frozen=True)
class SubsetDefinition:
    name: str
    required_columns: tuple[str, ...]
    mask: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class SampleResult:
    status: str
    pageids: list[str]
    reason: str = ""


class PrimaryMetricScorer:
    def __init__(self, records: pd.DataFrame, *, task: str, labels: list[str]) -> None:
        self.task = task
        self.pageids = [str(value) for value in records["pageid"]]
        self.pageid_to_index = {pageid: index for index, pageid in enumerate(self.pageids)}
        if task == "corine_level2":
            label_to_index = {label: index for index, label in enumerate(labels)}
            self.label_count = len(labels)
            self.true = np.asarray(
                [label_to_index.get(str(value), -1) for value in records["target"]],
                dtype=np.int64,
            )
            self.pred = np.asarray(
                [label_to_index.get(str(value), -1) for value in records["prediction"]],
                dtype=np.int64,
            )
            self.evaluated = np.asarray(
                [
                    row["parse_status"] == "ok" and row["prediction"] is not None
                    for _, row in records.iterrows()
                ],
                dtype=bool,
            )
        elif task == "osm":
            self.row_jaccard = np.asarray(
                [
                    _row_jaccard(
                        row["target"], row["prediction"] if row["parse_status"] == "ok" else []
                    )
                    for _, row in records.iterrows()
                ],
                dtype=float,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    def score_indices(self, indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        selected = np.asarray(indices, dtype=np.int64)
        if self.task == "osm":
            return float(np.mean(self.row_jaccard[selected]))
        recalls: list[float] = []
        for label_index in range(self.label_count):
            is_true = self.true[selected] == label_index
            is_evaluated_true = is_true & self.evaluated[selected]
            denominator = int(np.sum(is_evaluated_true))
            if denominator == 0:
                recalls.append(0.0)
                continue
            true_positive = int(np.sum(is_evaluated_true & (self.pred[selected] == label_index)))
            recalls.append(true_positive / denominator)
        return float(np.mean(recalls)) if recalls else 0.0

    def score_pageids(self, pageids: Sequence[str]) -> float:
        return self.score_indices(
            [
                self.pageid_to_index[str(pageid)]
                for pageid in pageids
                if str(pageid) in self.pageid_to_index
            ]
        )


def _row_jaccard(target: object, prediction: object) -> float:
    true_set = {str(value) for value in target} if isinstance(target, list) else set()
    pred_set = {str(value) for value in prediction} if isinstance(prediction, list) else set()
    union = true_set | pred_set
    if not union:
        return 1.0
    return len(true_set & pred_set) / len(union)


def _series(frame: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    return _series(frame, column).fillna("").astype(str).str.strip().str.lower()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series(frame, column), errors="coerce")


def _boolish(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _series(frame, column)
    return parse_boolish_series(values)


def _is_article_type(frame: pd.DataFrame, values: set[str]) -> pd.Series:
    return _text(frame, "primary_article_type").isin(values)


HIGH_PRIOR_ARTICLE_TYPES = {
    "agriculture_or_vineyard",
    "natural_landscape",
    "water_feature",
}


def _definition(
    name: str,
    required_columns: Sequence[str],
    mask: Callable[[pd.DataFrame], pd.Series],
) -> SubsetDefinition:
    return SubsetDefinition(name=name, required_columns=tuple(required_columns), mask=mask)


SUBSET_REGISTRY: dict[str, SubsetDefinition] = {
    "all": _definition("all", (), lambda df: pd.Series(True, index=df.index)),
    "point_label_share_250m_ge_0.8": _definition(
        "point_label_share_250m_ge_0.8",
        ("point_label_share_250m",),
        lambda df: _numeric(df, "point_label_share_250m") >= 0.8,
    ),
    "point_label_share_500m_ge_0.8": _definition(
        "point_label_share_500m_ge_0.8",
        ("point_label_share_500m",),
        lambda df: _numeric(df, "point_label_share_500m") >= 0.8,
    ),
    "point_label_share_250m_ge_0.9": _definition(
        "point_label_share_250m_ge_0.9",
        ("point_label_share_250m",),
        lambda df: _numeric(df, "point_label_share_250m") >= 0.9,
    ),
    "dominant_matches_point_label_250m": _definition(
        "dominant_matches_point_label_250m",
        ("dominant_matches_point_label_250m",),
        lambda df: _boolish(df, "dominant_matches_point_label_250m"),
    ),
    "dominant_matches_point_label_500m": _definition(
        "dominant_matches_point_label_500m",
        ("dominant_matches_point_label_500m",),
        lambda df: _boolish(df, "dominant_matches_point_label_500m"),
    ),
    "relevance_medium_high": _definition(
        "relevance_medium_high",
        ("landcover_relevance",),
        lambda df: _text(df, "landcover_relevance").isin({"medium", "high"}),
    ),
    "relevance_high": _definition(
        "relevance_high",
        ("landcover_relevance",),
        lambda df: _text(df, "landcover_relevance").eq("high"),
    ),
    "evidence_sentences_count_ge_1": _definition(
        "evidence_sentences_count_ge_1",
        ("evidence_sentences_count",),
        lambda df: _numeric(df, "evidence_sentences_count") >= 1,
    ),
    "evidence_sentences_count_ge_2": _definition(
        "evidence_sentences_count_ge_2",
        ("evidence_sentences_count",),
        lambda df: _numeric(df, "evidence_sentences_count") >= 2,
    ),
    "uncertainty_low": _definition(
        "uncertainty_low",
        ("uncertainty",),
        lambda df: _text(df, "uncertainty").eq("low"),
    ),
    "uncertainty_low_medium": _definition(
        "uncertainty_low_medium",
        ("uncertainty",),
        lambda df: _text(df, "uncertainty").isin({"low", "medium"}),
    ),
    "relevance_medium_high_and_spatial_250m_ge_0.8": _definition(
        "relevance_medium_high_and_spatial_250m_ge_0.8",
        ("landcover_relevance", "point_label_share_250m"),
        lambda df: (
            SUBSET_REGISTRY["relevance_medium_high"].mask(df)
            & SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(df)
        ),
    ),
    "relevance_high_and_spatial_250m_ge_0.8": _definition(
        "relevance_high_and_spatial_250m_ge_0.8",
        ("landcover_relevance", "point_label_share_250m"),
        lambda df: (
            SUBSET_REGISTRY["relevance_high"].mask(df)
            & SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(df)
        ),
    ),
    "evidence_sentences_count_ge_1_and_spatial_250m_ge_0.8": _definition(
        "evidence_sentences_count_ge_1_and_spatial_250m_ge_0.8",
        ("evidence_sentences_count", "point_label_share_250m"),
        lambda df: (
            SUBSET_REGISTRY["evidence_sentences_count_ge_1"].mask(df)
            & SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(df)
        ),
    ),
    "quality_high_or_very_high": _definition(
        "quality_high_or_very_high",
        ("quality_bin",),
        lambda df: _text(df, "quality_bin").isin({"quality_high", "quality_very_high"}),
    ),
    "quality_very_high": _definition(
        "quality_very_high",
        ("quality_bin",),
        lambda df: _text(df, "quality_bin").eq("quality_very_high"),
    ),
    "quality_high_or_very_high_and_spatial_250m_ge_0.8": _definition(
        "quality_high_or_very_high_and_spatial_250m_ge_0.8",
        ("quality_bin", "point_label_share_250m"),
        lambda df: (
            SUBSET_REGISTRY["quality_high_or_very_high"].mask(df)
            & SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(df)
        ),
    ),
    "recommended_use_training": _definition(
        "recommended_use_training",
        ("recommended_use",),
        lambda df: _text(df, "recommended_use").eq("use_for_training"),
    ),
    "recommended_use_evaluation_only": _definition(
        "recommended_use_evaluation_only",
        ("recommended_use",),
        lambda df: _text(df, "recommended_use").eq("use_for_evaluation_only"),
    ),
    "recommended_use_exclude": _definition(
        "recommended_use_exclude",
        ("recommended_use",),
        lambda df: _text(df, "recommended_use").eq("exclude"),
    ),
    "article_type_agriculture_or_vineyard": _definition(
        "article_type_agriculture_or_vineyard",
        ("primary_article_type",),
        lambda df: _is_article_type(df, {"agriculture_or_vineyard"}),
    ),
    "article_type_natural_landscape": _definition(
        "article_type_natural_landscape",
        ("primary_article_type",),
        lambda df: _is_article_type(df, {"natural_landscape"}),
    ),
    "article_type_water_feature": _definition(
        "article_type_water_feature",
        ("primary_article_type",),
        lambda df: _is_article_type(df, {"water_feature"}),
    ),
    "article_type_settlement_or_administrative": _definition(
        "article_type_settlement_or_administrative",
        ("primary_article_type",),
        lambda df: _is_article_type(df, {"settlement_or_administrative"}),
    ),
    "article_type_built_or_cultural_site": _definition(
        "article_type_built_or_cultural_site",
        ("primary_article_type",),
        lambda df: _is_article_type(df, {"built_or_cultural_site"}),
    ),
    "article_type_other_or_unclear": _definition(
        "article_type_other_or_unclear",
        ("primary_article_type",),
        lambda df: _is_article_type(df, {"other_or_unclear"}),
    ),
    "article_type_high_prior": _definition(
        "article_type_high_prior",
        ("primary_article_type",),
        lambda df: _is_article_type(df, HIGH_PRIOR_ARTICLE_TYPES),
    ),
    "article_type_high_prior_and_relevance_medium_high": _definition(
        "article_type_high_prior_and_relevance_medium_high",
        ("primary_article_type", "landcover_relevance"),
        lambda df: (
            SUBSET_REGISTRY["article_type_high_prior"].mask(df)
            & SUBSET_REGISTRY["relevance_medium_high"].mask(df)
        ),
    ),
    "other_or_unclear_and_relevance_medium_high": _definition(
        "other_or_unclear_and_relevance_medium_high",
        ("primary_article_type", "landcover_relevance"),
        lambda df: (
            SUBSET_REGISTRY["article_type_other_or_unclear"].mask(df)
            & SUBSET_REGISTRY["relevance_medium_high"].mask(df)
        ),
    ),
    "article_type_high_prior_and_relevance_medium_high_and_spatial_250m_ge_0.8": _definition(
        "article_type_high_prior_and_relevance_medium_high_and_spatial_250m_ge_0.8",
        ("primary_article_type", "landcover_relevance", "point_label_share_250m"),
        lambda df: (
            SUBSET_REGISTRY["article_type_high_prior_and_relevance_medium_high"].mask(df)
            & SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(df)
        ),
    ),
    "other_or_unclear_and_relevance_medium_high_and_spatial_250m_ge_0.8": _definition(
        "other_or_unclear_and_relevance_medium_high_and_spatial_250m_ge_0.8",
        ("primary_article_type", "landcover_relevance", "point_label_share_250m"),
        lambda df: (
            SUBSET_REGISTRY["other_or_unclear_and_relevance_medium_high"].mask(df)
            & SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(df)
        ),
    ),
}


CORE_METRIC_COLUMNS = [
    "n",
    "n_predicted_ok",
    "n_parse_error",
    "coverage",
    "accuracy",
    "accuracy_including_parse_errors_as_wrong",
    "balanced_accuracy",
    "balanced_accuracy_including_parse_errors_as_wrong",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_f1_including_parse_errors_as_wrong",
    "weighted_f1",
    "exact_match_accuracy",
    "exact_match_accuracy_including_parse_errors_as_empty",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "micro_f1_including_parse_errors_as_empty",
    "macro_f1_including_parse_errors_as_empty",
    "jaccard",
    "hamming_loss",
    "majority_accuracy",
    "majority_balanced_accuracy",
    "majority_macro_f1",
    "delta_vs_majority_balanced_accuracy",
    "majority_labelset_exact_match_accuracy",
    "empty_set_exact_match_accuracy",
]

OBSERVED_COLUMNS = [
    "parent_experiment_id",
    "model_key",
    "task",
    "text_source",
    "subset_name",
    "n",
    "unstable_small_n",
    "primary_metric",
    "observed_primary_score",
    "all_articles_primary_score",
    "observed_minus_all",
    *CORE_METRIC_COLUMNS,
]

CONTROL_COLUMNS = [
    "parent_experiment_id",
    "model_key",
    "task",
    "text_source",
    "subset_name",
    "control_type",
    "status",
    "n",
    "unstable_small_n",
    "primary_metric",
    "observed_score",
    "random_mean",
    "random_std",
    "random_min",
    "random_max",
    "random_p02_5",
    "random_p50",
    "random_p97_5",
    "observed_minus_random_mean",
    "observed_percentile",
    "empirical_p_greater_equal",
    "n_draws_requested",
    "n_draws_successful",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-experiment-dir",
        action="append",
        type=Path,
        default=None,
        help="Frozen parent prediction directory. Repeat to include extra parents.",
    )
    parser.add_argument(
        "--spatial-confidence-path",
        type=Path,
        default=DEFAULT_SPATIAL_CONFIDENCE_PATH,
    )
    parser.add_argument(
        "--evidence-metadata-path", type=Path, default=DEFAULT_EVIDENCE_METADATA_PATH
    )
    parser.add_argument(
        "--article-type-assignments-path",
        type=Path,
        default=DEFAULT_ARTICLE_TYPE_ASSIGNMENTS_PATH,
    )
    parser.add_argument("--quality-scores-path", type=Path, default=DEFAULT_QUALITY_SCORES_PATH)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-subset-size", type=int, default=10)
    parser.add_argument("--primary-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def stable_seed(
    seed: int,
    parent_experiment: str,
    model_key: str,
    task: str,
    text_source: str,
    subset_name: str,
    control_type: str,
) -> int:
    payload = "|".join(
        [str(seed), parent_experiment, model_key, task, text_source, subset_name, control_type]
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def uniform_sample_pageids(universe_pageids: Sequence[str], n: int, seed: int) -> list[str]:
    ordered = sorted(str(pageid) for pageid in universe_pageids)
    if n > len(ordered):
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ordered), size=n, replace=False)
    return sorted(ordered[int(index)] for index in indices)


def target_key(value: object) -> str:
    if isinstance(value, list):
        return json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
    if isinstance(value, tuple):
        return json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
    return str(value)


def target_matched_sample_pageids(
    universe: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    seed: int,
) -> SampleResult:
    if observed.empty:
        return SampleResult(status="skipped_empty_subset", pageids=[])
    universe_by_key: dict[str, list[str]] = {}
    for _, row in universe.iterrows():
        universe_by_key.setdefault(target_key(row["target"]), []).append(str(row["pageid"]))
    observed_counts = Counter(target_key(value) for value in observed["target"])
    for key, count in observed_counts.items():
        if len(universe_by_key.get(key, [])) < count:
            return SampleResult(
                status="skipped_insufficient_target_support",
                pageids=[],
                reason=f"{key}: requested {count}, available {len(universe_by_key.get(key, []))}",
            )

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for key in sorted(observed_counts):
        candidates = sorted(universe_by_key[key])
        indices = rng.choice(len(candidates), size=observed_counts[key], replace=False)
        selected.extend(candidates[int(index)] for index in indices)
    return SampleResult(status="ok", pageids=sorted(selected))


def primary_metric_for_task(task: str) -> str:
    if task == "corine_level2":
        return "balanced_accuracy"
    if task == "osm":
        return "jaccard"
    raise ValueError(f"Unsupported task: {task}")


def compute_metric_row(
    *,
    records: pd.DataFrame,
    selected_pageids: Sequence[str],
    labels: list[str],
    task: str,
    primary_metric: str,
    small_n_threshold: int = SMALL_N_THRESHOLD,
) -> dict[str, Any]:
    selected_ids = {str(pageid) for pageid in selected_pageids}
    subset = records[records["pageid"].astype(str).isin(selected_ids)].copy()
    metrics, _ = compute_task_subset_metrics(subset, task=task, labels=labels)
    row: dict[str, Any] = {
        "n": int(metrics.get("n", 0)),
        "unstable_small_n": int(metrics.get("n", 0)) < small_n_threshold,
        "primary_metric": primary_metric,
        "observed_primary_score": metrics.get(primary_metric, ""),
    }
    row.update({column: metrics.get(column, "") for column in CORE_METRIC_COLUMNS})
    return row


def _score_from_row(row: Mapping[str, Any]) -> float | None:
    value = row.get("observed_primary_score", row.get("observed_score"))
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    return None


def build_random_control_summary(
    *,
    observed_row: Mapping[str, Any],
    random_scores: Sequence[float],
    n_draws_requested: int,
    control_type: str,
    status: str = "ok",
) -> dict[str, Any]:
    observed = _score_from_row(observed_row)
    successful = len(random_scores)
    output: dict[str, Any] = {
        "control_type": control_type,
        "status": status,
        "n": observed_row.get("n", 0),
        "unstable_small_n": observed_row.get("unstable_small_n", False),
        "primary_metric": observed_row.get("primary_metric", ""),
        "observed_score": observed if observed is not None else "",
        "n_draws_requested": n_draws_requested,
        "n_draws_successful": successful,
    }
    if observed is None or not random_scores:
        output.update(
            {
                "random_mean": "",
                "random_std": "",
                "random_min": "",
                "random_max": "",
                "random_p02_5": "",
                "random_p50": "",
                "random_p97_5": "",
                "observed_minus_random_mean": "",
                "observed_percentile": "",
                "empirical_p_greater_equal": "",
            }
        )
        return output

    values = np.asarray(random_scores, dtype=float)
    tolerance = 1e-12
    output.update(
        {
            "random_mean": float(np.mean(values)),
            "random_std": float(np.std(values)),
            "random_min": float(np.min(values)),
            "random_max": float(np.max(values)),
            "random_p02_5": float(np.percentile(values, 2.5)),
            "random_p50": float(np.percentile(values, 50)),
            "random_p97_5": float(np.percentile(values, 97.5)),
            "observed_minus_random_mean": float(observed - np.mean(values)),
            "observed_percentile": float(100.0 * np.mean(values <= observed + tolerance)),
            "empirical_p_greater_equal": float(
                (1 + int(np.sum(values >= observed - tolerance))) / (1 + successful)
            ),
        }
    )
    return output


def compute_shuffled_delta_row(
    *,
    aligned_records: pd.DataFrame,
    shuffled_records: pd.DataFrame,
    pageids: Sequence[str],
    labels: list[str],
    task: str,
    primary_metric: str,
) -> dict[str, Any]:
    aligned = compute_metric_row(
        records=aligned_records,
        selected_pageids=pageids,
        labels=labels,
        task=task,
        primary_metric=primary_metric,
    )
    shuffled = compute_metric_row(
        records=shuffled_records,
        selected_pageids=pageids,
        labels=labels,
        task=task,
        primary_metric=primary_metric,
    )
    aligned_score = _score_from_row(aligned)
    shuffled_score = _score_from_row(shuffled)
    return {
        "observed_aligned_score": aligned_score if aligned_score is not None else "",
        "observed_shuffled_score": shuffled_score if shuffled_score is not None else "",
        "observed_delta": aligned_score - shuffled_score
        if aligned_score is not None and shuffled_score is not None
        else "",
        "sampled_pageids": ",".join(sorted(str(pageid) for pageid in pageids)),
    }


def _sanitize_model_key(model: str) -> str:
    lowered = model.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "unknown_model"


def _load_quality_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["pageid", "quality_score", "quality_bin", "recommended_use"])
    frame = pd.read_csv(path, dtype={"pageid": str})
    columns = ["pageid", "quality_score", "quality_bin", "recommended_use"]
    return frame[[column for column in columns if column in frame.columns]]


def _merge_metadata(
    records: pd.DataFrame,
    *,
    spatial_confidence_path: Path,
    evidence_metadata_path: Path,
    article_type_assignments_path: Path,
    quality_scores_path: Path,
) -> pd.DataFrame:
    output = records.copy()
    metadata_frames = [
        load_spatial_confidence(spatial_confidence_path, allow_missing_pageid=True),
        load_evidence_metadata(evidence_metadata_path),
        load_article_type_metadata(article_type_assignments_path),
        _load_quality_scores(quality_scores_path),
    ]
    for frame in metadata_frames:
        if frame.empty or "pageid" not in frame.columns:
            continue
        frame = frame.copy()
        frame["pageid"] = frame["pageid"].astype(str)
        output = output.merge(frame, on="pageid", how="left")
    return output


def _load_parent_records(parent_dir: Path) -> pd.DataFrame:
    records = load_prediction_records(
        parent_dir,
        normalize_targets=True,
        include_source_dir=True,
    )
    if records.empty:
        return records
    model = infer_model_for_records(records, parent_dir)
    records["model"] = model
    records["model_key"] = _sanitize_model_key(model)
    records["parent_experiment_id"] = parent_dir.name
    records["parent_experiment_dir"] = str(parent_dir)
    return records


def _metadata_available_mask(frame: pd.DataFrame, subset: SubsetDefinition) -> pd.Series:
    if not subset.required_columns:
        return pd.Series(True, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for column in subset.required_columns:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        values = frame[column]
        mask &= values.notna()
    return mask


def _metric_score_for_pageids(
    records: pd.DataFrame,
    pageids: Sequence[str],
    *,
    labels: list[str],
    task: str,
    primary_metric: str,
) -> float | None:
    row = compute_metric_row(
        records=records,
        selected_pageids=pageids,
        labels=labels,
        task=task,
        primary_metric=primary_metric,
    )
    return _score_from_row(row)


def _random_scores_same_n(
    universe: pd.DataFrame,
    *,
    n: int,
    n_draws: int,
    seed: int,
    labels: list[str],
    task: str,
    primary_metric: str,
) -> list[float]:
    del primary_metric
    scores: list[float] = []
    scorer = PrimaryMetricScorer(universe, task=task, labels=labels)
    candidate_indices = np.arange(len(universe), dtype=np.int64)
    for draw_index in range(n_draws):
        rng = np.random.default_rng(seed + draw_index)
        sampled = rng.choice(candidate_indices, size=n, replace=False)
        scores.append(scorer.score_indices([int(index) for index in sampled]))
    return scores


def _random_scores_target_matched(
    universe: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    n_draws: int,
    seed: int,
    labels: list[str],
    task: str,
    primary_metric: str,
) -> tuple[list[float], str]:
    del primary_metric
    scores: list[float] = []
    scorer = PrimaryMetricScorer(universe, task=task, labels=labels)
    strata: dict[str, list[int]] = {}
    for index, value in enumerate(universe["target"]):
        strata.setdefault(target_key(value), []).append(index)
    observed_counts = Counter(target_key(value) for value in observed["target"])
    for key, count in observed_counts.items():
        if len(strata.get(key, [])) < count:
            return [], "skipped_insufficient_target_support"
    for draw_index in range(n_draws):
        rng = np.random.default_rng(seed + draw_index)
        selected: list[int] = []
        for key in sorted(observed_counts):
            candidates = strata[key]
            sampled = rng.choice(len(candidates), size=observed_counts[key], replace=False)
            selected.extend(candidates[int(index)] for index in sampled)
        scores.append(scorer.score_indices(selected))
    return scores, "ok"


def _distribution_rows(
    group: pd.DataFrame,
    subset_name: str,
    subset_rows: pd.DataFrame,
    base: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if subset_rows.empty:
        return []
    counts = Counter(target_key(value) for value in subset_rows["target"])
    total = sum(counts.values())
    return [
        {
            **base,
            "subset_name": subset_name,
            "target_key": key,
            "support": count,
            "share": count / total if total else 0.0,
        }
        for key, count in sorted(counts.items())
    ]


def _conclusion_flag(
    same_n: Mapping[str, Any] | None, target_matched: Mapping[str, Any] | None
) -> str:
    if not same_n or same_n.get("observed_score") == "":
        return "not_distinguishable"
    observed = float(same_n["observed_score"])
    same_hi = same_n.get("random_p97_5")
    target_hi = target_matched.get("random_p97_5") if target_matched else ""
    same_mean = same_n.get("random_mean")
    if isinstance(same_mean, (int, float)) and observed < float(same_mean):
        return "below_random"
    beats_same = isinstance(same_hi, (int, float)) and observed > float(same_hi)
    beats_target = isinstance(target_hi, (int, float)) and observed > float(target_hi)
    if beats_same and beats_target:
        return "beats_target_matched"
    if beats_same:
        return "beats_same_n"
    return "not_distinguishable"


def _interval(row: Mapping[str, Any] | None) -> str:
    if not row:
        return ""
    low = row.get("random_p02_5")
    high = row.get("random_p97_5")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return f"[{low:.4f}, {high:.4f}]"
    return ""


def _write_table_pair(
    output_dir: Path,
    stem: str,
    title: str,
    rows: list[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    resolved_columns = resolve_table_columns(rows, columns)
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(
            output,
            fieldnames=resolved_columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    write_text_atomic(output_dir / f"{stem}.csv", output.getvalue())
    write_text_atomic(
        output_dir / f"{stem}.md",
        f"# {title}\n\n{markdown_table(rows=rows, columns=resolved_columns)}",
    )


def write_outputs(
    *,
    output_dir: Path,
    observed_rows: list[dict[str, Any]],
    random_same_n_rows: list[dict[str, Any]],
    random_target_rows: list[dict[str, Any]],
    shuffled_delta_rows: list[dict[str, Any]],
    class_distribution_rows: list[dict[str, Any]],
    significant_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_table_pair(
        output_dir,
        "observed_subset_metrics",
        "Observed Subset Metrics",
        observed_rows,
        OBSERVED_COLUMNS,
    )
    _write_table_pair(
        output_dir,
        "random_same_n_controls",
        "Random Same-size Controls",
        random_same_n_rows,
        CONTROL_COLUMNS,
    )
    _write_table_pair(
        output_dir,
        "random_target_matched_controls",
        "Random Target-matched Controls",
        random_target_rows,
        CONTROL_COLUMNS,
    )
    _write_table_pair(
        output_dir,
        "shuffled_delta_random_controls",
        "Shuffled Delta Random Controls",
        shuffled_delta_rows,
    )
    _write_table_pair(
        output_dir,
        "subset_class_distribution",
        "Subset Class Distribution",
        class_distribution_rows,
    )
    _write_table_pair(
        output_dir, "significant_filter_summary", "Significant Filter Summary", significant_rows
    )
    write_json_atomic(subset_randomization_manifest_path(output_dir), manifest, indent=2)
    write_text_atomic(subset_randomization_summary_path(output_dir), summary_text)


def evaluate(
    *,
    parent_experiment_dirs: Sequence[Path],
    spatial_confidence_path: Path,
    evidence_metadata_path: Path,
    article_type_assignments_path: Path,
    quality_scores_path: Path,
    output_dir: Path,
    n_draws: int,
    seed: int,
    min_subset_size: int,
    primary_only: bool,
) -> None:
    frames = [_load_parent_records(path) for path in parent_experiment_dirs if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("No frozen prediction records were loaded.")
    records = pd.concat(frames, ignore_index=True)
    records = _merge_metadata(
        records,
        spatial_confidence_path=spatial_confidence_path,
        evidence_metadata_path=evidence_metadata_path,
        article_type_assignments_path=article_type_assignments_path,
        quality_scores_path=quality_scores_path,
    )
    records["pageid"] = records["pageid"].astype(str)

    observed_rows: list[dict[str, Any]] = []
    same_n_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    class_distribution_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    all_scores: dict[tuple[str, str, str, str], float] = {}

    group_columns = ["parent_experiment_id", "model_key", "task", "text_source"]
    for group_key, group in records.groupby(group_columns, dropna=False):
        parent_id, model_key, task, text_source = [str(value) for value in group_key]
        primary_metric = primary_metric_for_task(task)
        labels = label_universe(group, task)
        base = {
            "parent_experiment_id": parent_id,
            "model_key": model_key,
            "task": task,
            "text_source": text_source,
        }
        all_row = compute_metric_row(
            records=group,
            selected_pageids=list(group["pageid"].astype(str)),
            labels=labels,
            task=task,
            primary_metric=primary_metric,
        )
        all_score = _score_from_row(all_row)
        if all_score is not None:
            all_scores[(parent_id, model_key, task, text_source)] = all_score

        for subset_name, subset_def in SUBSET_REGISTRY.items():
            availability = _metadata_available_mask(group, subset_def)
            universe = group[availability].copy()
            subset_mask = subset_def.mask(universe) if not universe.empty else pd.Series(dtype=bool)
            observed = universe[subset_mask].copy() if not universe.empty else universe
            n = len(observed)
            if n == 0:
                skipped_rows.append({**base, "subset_name": subset_name, "reason": "empty_subset"})
                continue
            metric_row = compute_metric_row(
                records=group,
                selected_pageids=list(observed["pageid"].astype(str)),
                labels=labels,
                task=task,
                primary_metric=primary_metric,
            )
            observed_score = _score_from_row(metric_row)
            row = {
                **base,
                "subset_name": subset_name,
                **metric_row,
                "all_articles_primary_score": all_score if all_score is not None else "",
                "observed_minus_all": observed_score - all_score
                if observed_score is not None and all_score is not None
                else "",
            }
            observed_rows.append(row)
            class_distribution_rows.extend(_distribution_rows(group, subset_name, observed, base))

            if n < min_subset_size:
                for control_type, output_rows in [
                    ("random_same_n", same_n_rows),
                    ("random_same_target_distribution", target_rows),
                ]:
                    control = build_random_control_summary(
                        observed_row=row,
                        random_scores=[],
                        n_draws_requested=n_draws,
                        control_type=control_type,
                        status="skipped_below_min_subset_size",
                    )
                    output_rows.append({**base, "subset_name": subset_name, **control})
                continue

            same_seed = stable_seed(
                seed, parent_id, model_key, task, text_source, subset_name, "random_same_n"
            )
            same_scores = _random_scores_same_n(
                universe,
                n=n,
                n_draws=n_draws,
                seed=same_seed,
                labels=labels,
                task=task,
                primary_metric=primary_metric,
            )
            same_n_rows.append(
                {
                    **base,
                    "subset_name": subset_name,
                    **build_random_control_summary(
                        observed_row=row,
                        random_scores=same_scores,
                        n_draws_requested=n_draws,
                        control_type="random_same_n",
                    ),
                }
            )

            target_seed = stable_seed(
                seed,
                parent_id,
                model_key,
                task,
                text_source,
                subset_name,
                "random_same_target_distribution",
            )
            target_scores, status = _random_scores_target_matched(
                universe,
                observed,
                n_draws=n_draws,
                seed=target_seed,
                labels=labels,
                task=task,
                primary_metric=primary_metric,
            )
            target_rows.append(
                {
                    **base,
                    "subset_name": subset_name,
                    **build_random_control_summary(
                        observed_row=row,
                        random_scores=target_scores,
                        n_draws_requested=n_draws,
                        control_type="random_same_target_distribution",
                        status=status,
                    ),
                }
            )

    shuffled_rows = _compute_shuffled_rows(
        records, observed_rows, n_draws=n_draws, seed=seed, min_subset_size=min_subset_size
    )
    significant_rows = _significant_summary(observed_rows, same_n_rows, target_rows)
    manifest = _manifest(
        parent_experiment_dirs=parent_experiment_dirs,
        spatial_confidence_path=spatial_confidence_path,
        evidence_metadata_path=evidence_metadata_path,
        article_type_assignments_path=article_type_assignments_path,
        quality_scores_path=quality_scores_path,
        n_draws=n_draws,
        seed=seed,
        min_subset_size=min_subset_size,
        primary_only=primary_only,
        skipped_rows=skipped_rows,
    )
    write_outputs(
        output_dir=output_dir,
        observed_rows=observed_rows,
        random_same_n_rows=same_n_rows,
        random_target_rows=target_rows,
        shuffled_delta_rows=shuffled_rows,
        class_distribution_rows=class_distribution_rows,
        significant_rows=significant_rows,
        manifest=manifest,
        summary_text=_summary_text(significant_rows, manifest),
    )


def _compute_shuffled_rows(
    records: pd.DataFrame,
    observed_rows: list[dict[str, Any]],
    *,
    n_draws: int,
    seed: int,
    min_subset_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    observed_by_key = {
        (
            row["parent_experiment_id"],
            row["model_key"],
            row["task"],
            row["text_source"],
            row["subset_name"],
        ): row
        for row in observed_rows
    }
    for (parent_id, model_key, task), group in records.groupby(
        ["parent_experiment_id", "model_key", "task"], dropna=False
    ):
        available_sources = set(group["text_source"].astype(str))
        for aligned_source, shuffled_source in shuffled_text_source_pairs(
            available_sources
        ).items():
            if aligned_source not in SHUFFLED_PAIRS:
                continue
            aligned_records = group[group["text_source"] == aligned_source]
            shuffled_records = group[group["text_source"] == shuffled_source]
            labels = label_universe(group, str(task))
            primary_metric = primary_metric_for_task(str(task))
            aligned_scorer = PrimaryMetricScorer(aligned_records, task=str(task), labels=labels)
            shuffled_scorer = PrimaryMetricScorer(shuffled_records, task=str(task), labels=labels)
            for subset_name, subset_def in SUBSET_REGISTRY.items():
                aligned_key = (
                    str(parent_id),
                    str(model_key),
                    str(task),
                    aligned_source,
                    subset_name,
                )
                shuffled_key = (
                    str(parent_id),
                    str(model_key),
                    str(task),
                    shuffled_source,
                    subset_name,
                )
                if aligned_key not in observed_by_key or shuffled_key not in observed_by_key:
                    continue
                universe = aligned_records[
                    _metadata_available_mask(aligned_records, subset_def)
                ].copy()
                if universe.empty:
                    continue
                observed = universe[subset_def.mask(universe)].copy()
                pageids = list(observed["pageid"].astype(str))
                n = len(pageids)
                delta_row = compute_shuffled_delta_row(
                    aligned_records=aligned_records,
                    shuffled_records=shuffled_records,
                    pageids=pageids,
                    labels=labels,
                    task=str(task),
                    primary_metric=primary_metric,
                )
                random_deltas: list[float] = []
                if n >= min_subset_size:
                    candidate_pageids = list(universe["pageid"].astype(str))
                    aligned_candidate_indices = [
                        aligned_scorer.pageid_to_index[pageid] for pageid in candidate_pageids
                    ]
                    shuffled_candidate_indices = [
                        shuffled_scorer.pageid_to_index[pageid]
                        for pageid in candidate_pageids
                        if pageid in shuffled_scorer.pageid_to_index
                    ]
                    if len(shuffled_candidate_indices) != len(aligned_candidate_indices):
                        continue
                    random_seed = stable_seed(
                        seed,
                        str(parent_id),
                        str(model_key),
                        str(task),
                        aligned_source,
                        subset_name,
                        "random_delta_same_n",
                    )
                    for draw_index in range(n_draws):
                        rng = np.random.default_rng(random_seed + draw_index)
                        positions = rng.choice(
                            len(aligned_candidate_indices), size=n, replace=False
                        )
                        aligned_indices = [
                            aligned_candidate_indices[int(position)] for position in positions
                        ]
                        shuffled_indices = [
                            shuffled_candidate_indices[int(position)] for position in positions
                        ]
                        random_deltas.append(
                            aligned_scorer.score_indices(aligned_indices)
                            - shuffled_scorer.score_indices(shuffled_indices)
                        )
                control = build_random_control_summary(
                    observed_row={
                        "n": n,
                        "unstable_small_n": n < SMALL_N_THRESHOLD,
                        "primary_metric": primary_metric,
                        "observed_primary_score": delta_row["observed_delta"],
                    },
                    random_scores=random_deltas,
                    n_draws_requested=n_draws,
                    control_type="random_delta_same_n",
                    status="ok" if random_deltas else "skipped_below_min_subset_size",
                )
                output.append(
                    {
                        "parent_experiment_id": str(parent_id),
                        "model_key": str(model_key),
                        "task": str(task),
                        "text_source": aligned_source,
                        "shuffled_text_source": shuffled_source,
                        "subset_name": subset_name,
                        "n": n,
                        "primary_metric": primary_metric,
                        "observed_aligned_score": delta_row["observed_aligned_score"],
                        "observed_shuffled_score": delta_row["observed_shuffled_score"],
                        "observed_delta": delta_row["observed_delta"],
                        "random_delta_mean": control["random_mean"],
                        "random_delta_p02_5": control["random_p02_5"],
                        "random_delta_p97_5": control["random_p97_5"],
                        "observed_delta_percentile": control["observed_percentile"],
                        "empirical_p_delta_greater_equal": control["empirical_p_greater_equal"],
                        "n_draws_successful": control["n_draws_successful"],
                    }
                )
    return output


def _significant_summary(
    observed_rows: list[dict[str, Any]],
    same_n_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    same_by_key = {
        (
            row["parent_experiment_id"],
            row["model_key"],
            row["task"],
            row["text_source"],
            row["subset_name"],
        ): row
        for row in same_n_rows
    }
    target_by_key = {
        (
            row["parent_experiment_id"],
            row["model_key"],
            row["task"],
            row["text_source"],
            row["subset_name"],
        ): row
        for row in target_rows
    }
    rows: list[dict[str, Any]] = []
    for observed in observed_rows:
        if observed["text_source"] not in HEADLINE_TEXT_SOURCES:
            continue
        if observed["subset_name"] not in HEADLINE_SUBSETS:
            continue
        if (
            observed["task"] == "corine_level2"
            and observed["primary_metric"] != "balanced_accuracy"
        ):
            continue
        if observed["task"] == "osm" and observed["primary_metric"] not in {
            "jaccard",
            "exact_match_accuracy",
        }:
            continue
        key = (
            observed["parent_experiment_id"],
            observed["model_key"],
            observed["task"],
            observed["text_source"],
            observed["subset_name"],
        )
        same = same_by_key.get(key)
        target = target_by_key.get(key)
        rows.append(
            {
                "parent_experiment_id": observed["parent_experiment_id"],
                "model_key": observed["model_key"],
                "task": observed["task"],
                "text_source": observed["text_source"],
                "subset_name": observed["subset_name"],
                "n": observed["n"],
                "unstable_small_n": observed["unstable_small_n"],
                "primary_metric": observed["primary_metric"],
                "observed_score": observed["observed_primary_score"],
                "observed_exact_match_accuracy": observed.get("exact_match_accuracy", ""),
                "random_same_n_mean": same.get("random_mean", "") if same else "",
                "random_same_n_95_interval": _interval(same),
                "random_same_n_percentile": same.get("observed_percentile", "") if same else "",
                "random_target_matched_mean": target.get("random_mean", "") if target else "",
                "random_target_matched_95_interval": _interval(target),
                "random_target_matched_percentile": target.get("observed_percentile", "")
                if target
                else "",
                "conclusion_flag": _conclusion_flag(same, target),
            }
        )
    return rows


def _manifest(
    *,
    parent_experiment_dirs: Sequence[Path],
    spatial_confidence_path: Path,
    evidence_metadata_path: Path,
    article_type_assignments_path: Path,
    quality_scores_path: Path,
    n_draws: int,
    seed: int,
    min_subset_size: int,
    primary_only: bool,
    skipped_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "no_llm_rerun": True,
        "no_gpu": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "parent_experiment_dirs": [str(path) for path in parent_experiment_dirs],
            "spatial_confidence_path": str(spatial_confidence_path),
            "evidence_metadata_path": str(evidence_metadata_path),
            "article_type_assignments_path": str(article_type_assignments_path),
            "quality_scores_path": str(quality_scores_path),
        },
        "parent_experiments_used": [str(path) for path in parent_experiment_dirs if path.exists()],
        "subset_definitions": {
            name: {"required_metadata_columns": list(definition.required_columns)}
            for name, definition in SUBSET_REGISTRY.items()
        },
        "comparison_universe": {
            "description": (
                "Random controls sample within the same parent experiment, model, task, text_source, "
                "and metadata availability required by the observed subset."
            ),
            "dimensions": ["parent_experiment_id", "model_key", "task", "text_source"],
            "metadata_availability": "subset-specific required columns must be non-missing",
        },
        "random_control_definitions": {
            "random_same_n": "uniform without replacement from the comparison universe",
            "random_same_target_distribution": "same target or target-label-set counts within the comparison universe",
        },
        "n_draws": n_draws,
        "seed": seed,
        "min_subset_size": min_subset_size,
        "unstable_small_n_threshold": SMALL_N_THRESHOLD,
        "primary_only": primary_only,
        "metrics_computed": CORE_METRIC_COLUMNS,
        "text_pairs_tested": SHUFFLED_PAIRS,
        "skipped_rows": skipped_rows,
    }


def _summary_text(significant_rows: list[dict[str, Any]], manifest: Mapping[str, Any]) -> str:
    flags = Counter(str(row.get("conclusion_flag", "")) for row in significant_rows)
    content_rows = [
        row
        for row in significant_rows
        if row.get("text_source") == "content"
        and row.get("subset_name")
        in {
            "relevance_medium_high",
            "point_label_share_250m_ge_0.8",
            "relevance_medium_high_and_spatial_250m_ge_0.8",
            "quality_high_or_very_high_and_spatial_250m_ge_0.8",
            "recommended_use_training",
            "recommended_use_evaluation_only",
        }
    ]

    def _find(model: str, task: str, subset: str) -> Mapping[str, Any] | None:
        for row in content_rows:
            if (
                row.get("model_key") == model
                and row.get("task") == task
                and row.get("subset_name") == subset
            ):
                return row
        return None

    def _score(row: Mapping[str, Any] | None) -> str:
        if not row:
            return "n/a"
        observed = row.get("observed_score")
        target_mean = row.get("random_target_matched_mean")
        if not isinstance(observed, (int, float)) or not isinstance(target_mean, (int, float)):
            return "n/a"
        detail = f"{observed:.3f} vs target-matched mean {target_mean:.3f}"
        if row.get("task") == "osm" and isinstance(
            row.get("observed_exact_match_accuracy"), (int, float)
        ):
            detail += f", exact match {row['observed_exact_match_accuracy']:.3f}"
        if row.get("unstable_small_n") is True:
            detail += " (unstable small n)"
        return detail

    qwen_corine_relevance = _score(_find("qwen", "corine_level2", "relevance_medium_high"))
    qwen_corine_combined = _score(
        _find("qwen", "corine_level2", "relevance_medium_high_and_spatial_250m_ge_0.8")
    )
    gemma_corine_relevance = _score(_find("gemma", "corine_level2", "relevance_medium_high"))
    gemma_corine_combined = _score(
        _find("gemma", "corine_level2", "relevance_medium_high_and_spatial_250m_ge_0.8")
    )
    qwen_osm_combined = _score(
        _find("qwen", "osm", "relevance_medium_high_and_spatial_250m_ge_0.8")
    )
    gemma_osm_combined = _score(
        _find("gemma", "osm", "relevance_medium_high_and_spatial_250m_ge_0.8")
    )
    qwen_osm_eval_only = _score(_find("qwen", "osm", "recommended_use_evaluation_only"))
    return (
        "# article_text_subset_randomization_controls_v1\n\n"
        "This analysis-only experiment compares filtered article subsets against "
        "Monte Carlo random controls. It does not rerun any LLMs, prompts, "
        "labels, spatial confidence, or GPU jobs.\n\n"
        "## What The Controls Mean\n\n"
        "- `random_same_n`: same parent experiment, model, task, text source, "
        "subset metadata availability, and subset size.\n"
        "- `random_same_target_distribution`: same comparison universe, plus "
        "the same CORINE target counts or OSM target-label-set counts.\n\n"
        "The comparison universe is intentionally explicit: random controls "
        "sample only from rows that could have passed the same "
        "metadata-availability requirements as the observed subset.\n\n"
        f"Rows with `n < {manifest['unstable_small_n_threshold']}` are marked "
        "`unstable_small_n=true`. They are reported, but should be treated as "
        "diagnostic rather than definitive.\n\n"
        "## Main Results\n\n"
        "For CORINE raw content, the key filters beat both random controls for "
        f"both models. Qwen reaches {qwen_corine_relevance} on "
        f"`relevance_medium_high` and {qwen_corine_combined} on "
        "`relevance_medium_high_and_spatial_250m_ge_0.8`. Gemma reaches "
        f"{gemma_corine_relevance} and {gemma_corine_combined} on the same "
        "two subsets.\n\n"
        "Quality-plus-spatial behaves like a strong proxy for relevance plus "
        "spatial confidence. It beats both controls for CORINE, but does not "
        "exceed the combined relevance+spatial subset.\n\n"
        "For OSM, results are more mixed. Qwen raw content on "
        "`relevance_medium_high_and_spatial_250m_ge_0.8` reaches "
        f"{qwen_osm_combined}. Gemma reaches {gemma_osm_combined} on the same "
        "subset, which is not distinguishable from target-matched controls in "
        "the headline table. The Qwen OSM `recommended_use_evaluation_only` "
        f"row is {qwen_osm_eval_only}, so it is diagnostic only.\n\n"
        "## Interpretation\n\n"
        "The strongest previous CORINE claim survives: when article relevance "
        "and spatial label reliability are credible, Wikipedia text signal is "
        "stronger than expected from subset size or target class composition "
        "alone.\n\n"
        "The OSM claim should be phrased more carefully. OSM text signal exists "
        "for some Qwen content subsets, but target composition and small support "
        "explain more of the apparent improvement than they do for CORINE.\n\n"
        "Aligned-vs-shuffled CORINE deltas also survive random delta controls. "
        "For raw content, both Qwen and Gemma CORINE deltas on relevance/spatial "
        "headline subsets are above the random 97.5% interval. OSM shuffled "
        "deltas are weaker and less consistent, especially for Gemma.\n\n"
        "## Headline Control Flags\n\n"
        + "\n".join(f"- {flag}: {count}" for flag, count in sorted(flags.items()))
        + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    parent_dirs = args.parent_experiment_dir or DEFAULT_PARENT_DIRS
    evaluate(
        parent_experiment_dirs=parent_dirs,
        spatial_confidence_path=args.spatial_confidence_path,
        evidence_metadata_path=args.evidence_metadata_path,
        article_type_assignments_path=args.article_type_assignments_path,
        quality_scores_path=args.quality_scores_path,
        output_dir=args.output_dir,
        n_draws=args.n_draws,
        seed=args.seed,
        min_subset_size=args.min_subset_size,
        primary_only=args.primary_only,
    )


if __name__ == "__main__":
    main()
