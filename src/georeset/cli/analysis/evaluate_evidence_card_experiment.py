"""Compare deterministic evidence-card classification outputs against prior sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.analysis.evaluation_metrics import (
    compute_task_subset_metrics,
)
from georeset.analysis.label_universe import label_universe
from georeset.analysis.pageid_frames import load_optional_pageid_csv
from georeset.analysis.prediction_loading import load_annotated_prediction_records
from georeset.analysis.shuffled_deltas import compute_shuffled_delta_rows
from georeset.text.evidence_cards import EVIDENCE_CARD_VERSION
from georeset.utils.json_io import (
    write_dict_rows_table_pair_atomic,
    write_json_atomic,
    write_text_atomic,
)

DEFAULT_EVIDENCE_CARD_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_evidence_card_v1__qwen3_6_27b_q4_0"
)
DEFAULT_PREVIOUS_QWEN_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_classification_e2e_with_shuffled_control_v1"
)
DEFAULT_LANDUSE_EVIDENCE_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_classification_landuse_evidence_v1__qwen3_6_27b_q4_0"
)
DEFAULT_QUALITY_SCORES_PATH = Path(
    "data/experiments/article_text_supervision_quality_score_v1/quality_scores.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/experiments/evidence_card_comparison_v1")
DEFAULT_EXPERIMENT_ID = "evidence_card_comparison_v1"

EVIDENCE_CARD_SOURCES = {
    "evidence_card",
    "evidence_card_shuffled",
    "content_with_evidence_card",
    "content_with_evidence_card_shuffled",
}
PREVIOUS_SOURCES = {"summary", "summary_no_place", "content"}
LANDUSE_SOURCES = {"landuse_evidence_summary"}
SHUFFLED_PAIRS = {
    "evidence_card": "evidence_card_shuffled",
    "content_with_evidence_card": "content_with_evidence_card_shuffled",
}
OUTPUT_COLUMNS = [
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
    "weighted_f1",
    "exact_match_accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "jaccard",
    "hamming_loss",
    "majority_accuracy",
    "majority_balanced_accuracy",
    "majority_macro_f1",
    "delta_vs_majority_balanced_accuracy",
    "majority_labelset_exact_match_accuracy",
    "empty_set_exact_match_accuracy",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-card-experiment-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_CARD_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--previous-qwen-experiment-dir",
        type=Path,
        default=DEFAULT_PREVIOUS_QWEN_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--landuse-evidence-experiment-dir",
        type=Path,
        default=DEFAULT_LANDUSE_EVIDENCE_EXPERIMENT_DIR,
    )
    parser.add_argument("--quality-scores-path", type=Path, default=DEFAULT_QUALITY_SCORES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args(argv)


def _load_records(
    experiment_dir: Path,
    text_sources: set[str],
    source_group: str,
) -> pd.DataFrame:
    return load_annotated_prediction_records(
        experiment_dir,
        text_sources=text_sources,
        source_group=source_group,
    )


def _metric_row(records: pd.DataFrame, subset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = str(records["task"].iloc[0])
    labels = label_universe(records, task)
    metrics, per_class = compute_task_subset_metrics(records, task=task, labels=labels)
    row = {
        "subset": subset,
        "model": records["model"].iloc[0],
        "task": task,
        "text_source": records["text_source"].iloc[0],
    }
    row.update(metrics)
    return row, [
        {
            "subset": subset,
            "model": records["model"].iloc[0],
            "task": task,
            "text_source": records["text_source"].iloc[0],
            **item,
        }
        for item in per_class
    ]


def _subset_masks(records: pd.DataFrame) -> dict[str, pd.Series]:
    empty = pd.Series("", index=records.index)
    relevance = records.get("landcover_relevance", empty).astype("string").str.lower()
    spatial = pd.to_numeric(records.get("point_label_share_250m", empty), errors="coerce")
    quality_bin = records.get("quality_bin", empty).astype("string")
    recommended_use = records.get("recommended_use", empty).astype("string")
    high_quality = quality_bin.isin(["quality_high", "quality_very_high"])
    return {
        "all": pd.Series(True, index=records.index),
        "relevance_medium_high": relevance.isin(["medium", "high"]),
        "spatial_250m_ge_0.8": spatial >= 0.8,
        "relevance_medium_high_and_spatial_250m_ge_0.8": relevance.isin(["medium", "high"])
        & (spatial >= 0.8),
        "quality_high_or_very_high": high_quality,
        "quality_high_or_very_high_and_spatial_250m_ge_0.8": high_quality & (spatial >= 0.8),
        "recommended_use_training": recommended_use == "use_for_training",
        "recommended_use_evaluation_only": recommended_use == "use_for_evaluation_only",
        "recommended_use_exclude": recommended_use == "exclude",
    }


def _compute_rows(records: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overview_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    masks = _subset_masks(records)
    for subset, mask in masks.items():
        subset_records = records[mask].copy()
        if subset_records.empty:
            continue
        for _, group in subset_records.groupby(["model", "task", "text_source"], sort=True):
            row, per_class = _metric_row(group, subset)
            overview_rows.append(row)
            per_class_rows.extend(per_class)
    return overview_rows, per_class_rows


def _shuffled_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compute_shuffled_delta_rows(
        rows,
        shuffled_pairs=SHUFFLED_PAIRS,
        model_columns=("model",),
    )


def evaluate(
    *,
    evidence_card_experiment_dir: Path,
    previous_qwen_experiment_dir: Path,
    landuse_evidence_experiment_dir: Path,
    quality_scores_path: Path,
    output_dir: Path,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    frames = [
        _load_records(evidence_card_experiment_dir, EVIDENCE_CARD_SOURCES, "evidence_card"),
        _load_records(previous_qwen_experiment_dir, PREVIOUS_SOURCES, "previous_qwen"),
        _load_records(landuse_evidence_experiment_dir, LANDUSE_SOURCES, "landuse_evidence"),
    ]
    records = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if records.empty:
        raise ValueError("No prediction records were loaded.")

    quality_scores = load_optional_pageid_csv(quality_scores_path)
    if not quality_scores.empty:
        records = records.merge(quality_scores, on="pageid", how="left")

    overview_rows, per_class_rows = _compute_rows(records)
    shuffled_delta_rows = _shuffled_deltas(overview_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_subset_rows = [row for row in overview_rows if row["subset"] == "all"]
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_card_vs_previous_sources",
        title="Evidence Card vs Previous Sources",
        rows=all_subset_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_card_quality_subsets",
        title="Evidence Card Quality Subsets",
        rows=overview_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_card_shuffled_deltas",
        title="Evidence Card Shuffled Deltas",
        rows=shuffled_delta_rows,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_card_per_class_corine",
        title="Evidence Card CORINE Per-class Metrics",
        rows=per_class_rows,
    )
    osm_rows = [row for row in overview_rows if row["task"] == "osm"]
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_card_osm_metrics",
        title="Evidence Card OSM Metrics",
        rows=osm_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "no_llm_card_generation": True,
            "no_summarization_rerun": True,
            "deterministic_card_version": EVIDENCE_CARD_VERSION,
            "input_paths": {
                "evidence_card_experiment_dir": str(evidence_card_experiment_dir),
                "previous_qwen_experiment_dir": str(previous_qwen_experiment_dir),
                "landuse_evidence_experiment_dir": str(landuse_evidence_experiment_dir),
                "quality_scores_path": str(quality_scores_path),
            },
            "text_sources_compared": sorted(set(records["text_source"])),
            "required_baselines": ["summary", "summary_no_place", "content", "landuse_evidence_summary"],
            "subsets": sorted({row["subset"] for row in overview_rows}),
            "metrics": OUTPUT_COLUMNS,
        },
        indent=2,
    )
    write_text_atomic(
        output_dir / "summary.md",
        "# evidence_card_comparison_v1\n\n"
        "This analysis compares deterministic evidence-card text sources against prior Qwen "
        "summary, no-place summary, raw-content, and land-use evidence-summary baselines. "
        "`content_with_evidence_card` preserves the original raw article text, so it is not "
        "a no-place text source.\n",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        evidence_card_experiment_dir=args.evidence_card_experiment_dir,
        previous_qwen_experiment_dir=args.previous_qwen_experiment_dir,
        landuse_evidence_experiment_dir=args.landuse_evidence_experiment_dir,
        quality_scores_path=args.quality_scores_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
