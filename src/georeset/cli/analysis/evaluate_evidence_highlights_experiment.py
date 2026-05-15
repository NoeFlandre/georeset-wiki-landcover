"""Compare deterministic evidence-highlighted content against prior text sources."""

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
from georeset.analysis.quality_subsets import quality_subset_masks
from georeset.analysis.shuffled_deltas import compute_shuffled_delta_rows
from georeset.classification.text_sources import shuffled_text_source_pairs
from georeset.text.evidence_highlights import EVIDENCE_HIGHLIGHTS_VERSION
from georeset.utils.json_io import (
    write_dict_rows_table_pair_atomic,
    write_json_atomic,
    write_text_atomic,
)

DEFAULT_QWEN_HIGHLIGHTS_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_evidence_highlights_v1__qwen3_6_27b_q4_0"
)
DEFAULT_GEMMA_HIGHLIGHTS_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_evidence_highlights_v1__gemma4_31b_it_q4_0"
)
DEFAULT_QWEN_PREVIOUS_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_classification_e2e_with_shuffled_control_v1"
)
DEFAULT_GEMMA_PREVIOUS_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
)
DEFAULT_QWEN_LANDUSE_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_classification_landuse_evidence_v1__qwen3_6_27b_q4_0"
)
DEFAULT_GEMMA_LANDUSE_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_classification_landuse_evidence_v1__gemma4_31b_it_q4_0"
)
DEFAULT_QWEN_EVIDENCE_CARD_EXPERIMENT_DIR = Path(
    "data/experiments/article_text_evidence_card_v1__qwen3_6_27b_q4_0"
)
DEFAULT_QUALITY_SCORES_PATH = Path(
    "data/experiments/article_text_supervision_quality_score_v1/quality_scores.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/experiments/evidence_highlights_comparison_v1")
DEFAULT_EXPERIMENT_ID = "evidence_highlights_comparison_v1"

HIGHLIGHT_SOURCES = {
    "content_with_evidence_highlights",
    "content_with_evidence_highlights_shuffled",
}
PREVIOUS_SOURCES = {"summary", "summary_no_place", "content", "content_shuffled"}
LANDUSE_SOURCES = {"landuse_evidence_summary", "landuse_evidence_summary_shuffled"}
QWEN_CARD_SOURCES = {"content_with_evidence_card", "content_with_evidence_card_shuffled"}
OUTPUT_COLUMNS = [
    "subset",
    "model_key",
    "model",
    "task",
    "text_source",
    "source_group",
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
    "delta_vs_majority_accuracy",
    "delta_vs_majority_balanced_accuracy",
    "delta_vs_majority_macro_f1",
    "majority_labelset_exact_match_accuracy",
    "empty_set_exact_match_accuracy",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-evidence-highlights-experiment-dir",
        type=Path,
        default=DEFAULT_QWEN_HIGHLIGHTS_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--gemma-evidence-highlights-experiment-dir",
        type=Path,
        default=DEFAULT_GEMMA_HIGHLIGHTS_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--qwen-previous-experiment-dir",
        type=Path,
        default=DEFAULT_QWEN_PREVIOUS_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--gemma-previous-experiment-dir",
        type=Path,
        default=DEFAULT_GEMMA_PREVIOUS_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--qwen-landuse-evidence-experiment-dir",
        type=Path,
        default=DEFAULT_QWEN_LANDUSE_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--gemma-landuse-evidence-experiment-dir",
        type=Path,
        default=DEFAULT_GEMMA_LANDUSE_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--qwen-evidence-card-experiment-dir",
        type=Path,
        default=DEFAULT_QWEN_EVIDENCE_CARD_EXPERIMENT_DIR,
    )
    parser.add_argument("--quality-scores-path", type=Path, default=DEFAULT_QUALITY_SCORES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    return parser.parse_args(argv)


def _load_records(
    experiment_dir: Path,
    text_sources: set[str],
    source_group: str,
    model_key: str,
) -> pd.DataFrame:
    return load_annotated_prediction_records(
        experiment_dir,
        text_sources=text_sources,
        source_group=source_group,
        model_key=model_key,
    )


def _metric_row(records: pd.DataFrame, subset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = str(records["task"].iloc[0])
    labels = label_universe(records, task)
    metrics, per_class = compute_task_subset_metrics(records, task=task, labels=labels)
    row = {
        "subset": subset,
        "model_key": records["model_key"].iloc[0],
        "model": records["model"].iloc[0],
        "task": task,
        "text_source": records["text_source"].iloc[0],
        "source_group": records["source_group"].iloc[0],
    }
    row.update(metrics)
    return row, [
        {
            "subset": subset,
            "model_key": records["model_key"].iloc[0],
            "model": records["model"].iloc[0],
            "task": task,
            "text_source": records["text_source"].iloc[0],
            **item,
        }
        for item in per_class
    ]


def _subset_masks(records: pd.DataFrame) -> dict[str, pd.Series]:
    return quality_subset_masks(records)


def _compute_rows(records: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overview_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for subset, mask in _subset_masks(records).items():
        subset_records = records[mask].copy()
        if subset_records.empty:
            continue
        for _, group in subset_records.groupby(["model_key", "task", "text_source"], sort=True):
            row, per_class = _metric_row(group, subset)
            overview_rows.append(row)
            per_class_rows.extend(per_class)
    return overview_rows, per_class_rows


def _shuffled_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return compute_shuffled_delta_rows(
        rows,
        shuffled_pairs=shuffled_text_source_pairs({str(row["text_source"]) for row in rows}),
        model_columns=("model_key", "model"),
    )


def evaluate(
    *,
    qwen_evidence_highlights_experiment_dir: Path,
    gemma_evidence_highlights_experiment_dir: Path,
    qwen_previous_experiment_dir: Path,
    gemma_previous_experiment_dir: Path,
    qwen_landuse_evidence_experiment_dir: Path,
    gemma_landuse_evidence_experiment_dir: Path,
    qwen_evidence_card_experiment_dir: Path,
    quality_scores_path: Path,
    output_dir: Path,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    frames = [
        _load_records(
            qwen_evidence_highlights_experiment_dir,
            HIGHLIGHT_SOURCES,
            "evidence_highlights",
            "qwen",
        ),
        _load_records(
            gemma_evidence_highlights_experiment_dir,
            HIGHLIGHT_SOURCES,
            "evidence_highlights",
            "gemma4_31b_it_q4_0",
        ),
        _load_records(qwen_previous_experiment_dir, PREVIOUS_SOURCES, "previous", "qwen"),
        _load_records(
            gemma_previous_experiment_dir,
            PREVIOUS_SOURCES,
            "previous",
            "gemma4_31b_it_q4_0",
        ),
        _load_records(
            qwen_landuse_evidence_experiment_dir,
            LANDUSE_SOURCES,
            "landuse_evidence",
            "qwen",
        ),
        _load_records(
            gemma_landuse_evidence_experiment_dir,
            LANDUSE_SOURCES,
            "landuse_evidence",
            "gemma4_31b_it_q4_0",
        ),
        _load_records(
            qwen_evidence_card_experiment_dir,
            QWEN_CARD_SOURCES,
            "evidence_card",
            "qwen",
        ),
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
        stem="evidence_highlights_vs_previous_sources",
        title="Evidence Highlights vs Previous Sources",
        rows=all_subset_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_highlights_quality_subsets",
        title="Evidence Highlights Quality Subsets",
        rows=overview_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_highlights_shuffled_deltas",
        title="Evidence Highlights Shuffled Deltas",
        rows=shuffled_delta_rows,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_highlights_per_class_corine",
        title="Evidence Highlights CORINE Per-class Metrics",
        rows=per_class_rows,
    )
    osm_rows = [row for row in overview_rows if row["task"] == "osm"]
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="evidence_highlights_osm_metrics",
        title="Evidence Highlights OSM Metrics",
        rows=osm_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "no_llm_highlight_generation": True,
            "no_summarization_rerun": True,
            "deterministic_highlight_version": EVIDENCE_HIGHLIGHTS_VERSION,
            "input_paths": {
                "qwen_evidence_highlights_experiment_dir": str(
                    qwen_evidence_highlights_experiment_dir
                ),
                "gemma_evidence_highlights_experiment_dir": str(
                    gemma_evidence_highlights_experiment_dir
                ),
                "qwen_previous_experiment_dir": str(qwen_previous_experiment_dir),
                "gemma_previous_experiment_dir": str(gemma_previous_experiment_dir),
                "qwen_landuse_evidence_experiment_dir": str(
                    qwen_landuse_evidence_experiment_dir
                ),
                "gemma_landuse_evidence_experiment_dir": str(
                    gemma_landuse_evidence_experiment_dir
                ),
                "qwen_evidence_card_experiment_dir": str(qwen_evidence_card_experiment_dir),
                "quality_scores_path": str(quality_scores_path),
            },
            "text_sources_compared": sorted(set(records["text_source"])),
            "subsets": sorted({row["subset"] for row in overview_rows}),
            "metrics": OUTPUT_COLUMNS,
        },
        indent=2,
    )
    write_text_atomic(
        output_dir / "summary.md",
        "# evidence_highlights_comparison_v1\n\n"
        "This analysis compares deterministic evidence-highlighted raw content against "
        "previous summary, raw-content, land-use evidence-summary, and available "
        "evidence-card baselines. The highlighted source preserves raw article text, "
        "so it is not a no-place text source.\n",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        qwen_evidence_highlights_experiment_dir=args.qwen_evidence_highlights_experiment_dir,
        gemma_evidence_highlights_experiment_dir=args.gemma_evidence_highlights_experiment_dir,
        qwen_previous_experiment_dir=args.qwen_previous_experiment_dir,
        gemma_previous_experiment_dir=args.gemma_previous_experiment_dir,
        qwen_landuse_evidence_experiment_dir=args.qwen_landuse_evidence_experiment_dir,
        gemma_landuse_evidence_experiment_dir=args.gemma_landuse_evidence_experiment_dir,
        qwen_evidence_card_experiment_dir=args.qwen_evidence_card_experiment_dir,
        quality_scores_path=args.quality_scores_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
