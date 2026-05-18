"""Compare retrieved evidence windows against full, random, and shuffled text."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.analysis.evaluation_metrics import compute_task_subset_metrics
from georeset.analysis.label_universe import label_universe
from georeset.analysis.pageid_frames import load_optional_pageid_csv
from georeset.analysis.prediction_loading import load_annotated_prediction_records
from georeset.analysis.quality_subsets import quality_subset_masks
from georeset.analysis.shuffled_deltas import compute_shuffled_delta_rows, primary_metric_name
from georeset.classification.text_sources import shuffled_text_source_pairs
from georeset.experiment_paths import experiment_artifact_dir, experiment_artifact_file
from georeset.text.retrieved_evidence_windows import RETRIEVED_EVIDENCE_WINDOWS_VERSION
from georeset.utils.json_io import (
    write_dict_rows_table_pair_atomic,
    write_json_atomic,
    write_text_atomic,
)

DEFAULT_QWEN_RETRIEVED_EXPERIMENT_DIR = experiment_artifact_dir(
    "article_text_retrieved_evidence_windows_v1__qwen3_6_27b_q4_0"
)
DEFAULT_GEMMA_RETRIEVED_EXPERIMENT_DIR = experiment_artifact_dir(
    "article_text_retrieved_evidence_windows_v1__gemma4_31b_it_q4_0"
)
DEFAULT_QWEN_PREVIOUS_EXPERIMENT_DIR = experiment_artifact_dir(
    "article_text_classification_e2e_with_shuffled_control_v1"
)
DEFAULT_GEMMA_PREVIOUS_EXPERIMENT_DIR = experiment_artifact_dir(
    "article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
)
DEFAULT_QUALITY_SCORES_PATH = experiment_artifact_file(
    "article_text_supervision_quality_score_v1", "quality_scores.csv"
)
DEFAULT_EXPERIMENT_ID = "retrieved_evidence_windows_comparison_v1"
DEFAULT_OUTPUT_DIR = experiment_artifact_dir(DEFAULT_EXPERIMENT_ID)

RETRIEVED_SOURCES = {
    "retrieved_evidence_windows",
    "retrieved_evidence_sentences_only",
    "random_sentence_windows",
    "retrieved_evidence_windows_no_place",
    "retrieved_evidence_windows_shuffled",
}
PREVIOUS_SOURCES = {"content", "content_shuffled"}
PAIRWISE_COMPARISONS = {
    "content_minus_retrieved": ("content", "retrieved_evidence_windows"),
    "retrieved_minus_random": ("retrieved_evidence_windows", "random_sentence_windows"),
    "retrieved_minus_sentences_only": (
        "retrieved_evidence_windows",
        "retrieved_evidence_sentences_only",
    ),
    "retrieved_minus_no_place": (
        "retrieved_evidence_windows",
        "retrieved_evidence_windows_no_place",
    ),
}
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
        "--qwen-retrieved-experiment-dir",
        type=Path,
        default=DEFAULT_QWEN_RETRIEVED_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--gemma-retrieved-experiment-dir",
        type=Path,
        default=DEFAULT_GEMMA_RETRIEVED_EXPERIMENT_DIR,
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


def _compute_rows(records: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overview_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for subset, mask in quality_subset_masks(records).items():
        subset_records = records[mask].copy()
        if subset_records.empty:
            continue
        for _, group in subset_records.groupby(["model_key", "task", "text_source"], sort=True):
            row, per_class = _metric_row(group, subset)
            overview_rows.append(row)
            per_class_rows.extend(per_class)
    return overview_rows, per_class_rows


def _pairwise_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["subset"], row["model_key"], row["model"], row["task"], row["text_source"]): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        for comparison, (source_a, source_b) in PAIRWISE_COMPARISONS.items():
            if row["text_source"] != source_a:
                continue
            other = by_key.get(
                (row["subset"], row["model_key"], row["model"], row["task"], source_b)
            )
            if other is None:
                continue
            metric = primary_metric_name(str(row["task"]))
            score_a = float(row.get(metric, 0.0) or 0.0)
            score_b = float(other.get(metric, 0.0) or 0.0)
            output.append(
                {
                    "subset": row["subset"],
                    "model_key": row["model_key"],
                    "model": row["model"],
                    "task": row["task"],
                    "comparison": comparison,
                    "metric": metric,
                    "source_a": source_a,
                    "source_b": source_b,
                    "n_a": row.get("n"),
                    "n_b": other.get("n"),
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta": score_a - score_b,
                }
            )
    return output


def _agreement_rows(records: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ok = records[records["parse_status"] == "ok"].copy()
    for subset, mask in quality_subset_masks(ok).items():
        subset_records = ok[mask].copy()
        if subset_records.empty:
            continue
        for (task, text_source), group in subset_records.groupby(
            ["task", "text_source"], sort=True
        ):
            if task != "corine_level2":
                continue
            pivot = group.pivot_table(
                index=["pageid", "target"],
                columns="model_key",
                values="prediction",
                aggfunc="first",
            ).reset_index()
            if not {"qwen", "gemma4_31b_it_q4_0"}.issubset(pivot.columns):
                continue
            agree = pivot["qwen"] == pivot["gemma4_31b_it_q4_0"]
            agree_count = int(agree.sum())
            correct_agree = agree & (pivot["qwen"].astype(str) == pivot["target"].astype(str))
            output.append(
                {
                    "subset": subset,
                    "task": task,
                    "text_source": text_source,
                    "n": len(pivot),
                    "agreement_count": agree_count,
                    "agreement_rate": agree_count / len(pivot) if len(pivot) else 0.0,
                    "agreement_precision_vs_target": (
                        int(correct_agree.sum()) / agree_count if agree_count else 0.0
                    ),
                }
            )
    return output


def evaluate(
    *,
    qwen_retrieved_experiment_dir: Path,
    gemma_retrieved_experiment_dir: Path,
    qwen_previous_experiment_dir: Path,
    gemma_previous_experiment_dir: Path,
    quality_scores_path: Path,
    output_dir: Path,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    frames = [
        _load_records(qwen_retrieved_experiment_dir, RETRIEVED_SOURCES, "retrieved", "qwen"),
        _load_records(
            gemma_retrieved_experiment_dir,
            RETRIEVED_SOURCES,
            "retrieved",
            "gemma4_31b_it_q4_0",
        ),
        _load_records(qwen_previous_experiment_dir, PREVIOUS_SOURCES, "previous", "qwen"),
        _load_records(
            gemma_previous_experiment_dir,
            PREVIOUS_SOURCES,
            "previous",
            "gemma4_31b_it_q4_0",
        ),
    ]
    records = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if records.empty:
        raise ValueError("No prediction records were loaded.")

    quality_scores = load_optional_pageid_csv(quality_scores_path)
    if not quality_scores.empty:
        records = records.merge(quality_scores, on="pageid", how="left")

    overview_rows, per_class_rows = _compute_rows(records)
    shuffled_delta_rows = compute_shuffled_delta_rows(
        overview_rows,
        shuffled_pairs=shuffled_text_source_pairs(
            {str(row["text_source"]) for row in overview_rows}
        ),
        model_columns=("model_key", "model"),
    )
    pairwise_rows = _pairwise_deltas(overview_rows)
    agreement_rows = _agreement_rows(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="retrieved_evidence_windows_quality_subsets",
        title="Retrieved Evidence Windows Quality Subsets",
        rows=overview_rows,
        columns=OUTPUT_COLUMNS,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="retrieved_evidence_windows_shuffled_deltas",
        title="Retrieved Evidence Windows Shuffled Deltas",
        rows=shuffled_delta_rows,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="retrieved_evidence_windows_pairwise_deltas",
        title="Retrieved Evidence Windows Pairwise Deltas",
        rows=pairwise_rows,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="retrieved_evidence_windows_model_agreement",
        title="Retrieved Evidence Windows Model Agreement",
        rows=agreement_rows,
    )
    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="retrieved_evidence_windows_per_class_corine",
        title="Retrieved Evidence Windows CORINE Per-class Metrics",
        rows=per_class_rows,
    )
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "deterministic_retrieval_version": RETRIEVED_EVIDENCE_WINDOWS_VERSION,
            "input_paths": {
                "qwen_retrieved_experiment_dir": str(qwen_retrieved_experiment_dir),
                "gemma_retrieved_experiment_dir": str(gemma_retrieved_experiment_dir),
                "qwen_previous_experiment_dir": str(qwen_previous_experiment_dir),
                "gemma_previous_experiment_dir": str(gemma_previous_experiment_dir),
                "quality_scores_path": str(quality_scores_path),
            },
        },
        indent=2,
        ensure_ascii=False,
    )
    write_text_atomic(
        output_dir / "summary.md",
        "# retrieved_evidence_windows_comparison_v1\n\n"
        "Analysis tables comparing retrieved evidence windows, full content, random windows, "
        "sentence-only retrieval, no-place retrieval, and shuffled controls.\n",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        qwen_retrieved_experiment_dir=args.qwen_retrieved_experiment_dir,
        gemma_retrieved_experiment_dir=args.gemma_retrieved_experiment_dir,
        qwen_previous_experiment_dir=args.qwen_previous_experiment_dir,
        gemma_previous_experiment_dir=args.gemma_previous_experiment_dir,
        quality_scores_path=args.quality_scores_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
