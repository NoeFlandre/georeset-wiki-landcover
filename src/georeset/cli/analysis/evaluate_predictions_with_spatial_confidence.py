"""Reevaluate frozen classification predictions on CORINE spatial-confidence subsets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.analysis.evaluation_metrics import (
    compute_multilabel_subset_metrics,
    compute_single_label_subset_metrics,
)
from georeset.analysis.label_universe import label_universe
from georeset.analysis.prediction_loading import load_prediction_records
from georeset.analysis.spatial_confidence_loading import load_spatial_confidence
from georeset.classification.labels import CORINE_LEVEL2_DESCRIPTIONS
from georeset.experiment_paths import experiment_artifact_dir, experiment_artifact_file
from georeset.utils.boolish import parse_boolish_series
from georeset.utils.json_io import (
    write_dict_rows_csv_atomic,
    write_dict_rows_table_pair_atomic,
    write_json_atomic,
    write_text_atomic,
)
from georeset.utils.math import safe_div

EXPERIMENT_ID = "article_text_classification_spatial_confidence_v1"
PARENT_EXPERIMENT_ID = "article_text_classification_e2e_with_shuffled_control_v1"
SPATIAL_EXPERIMENT_ID = "corine_spatial_confidence_v1"
DEFAULT_PARENT_DIR = Path(experiment_artifact_dir(PARENT_EXPERIMENT_ID))
DEFAULT_SPATIAL_PATH = experiment_artifact_file(SPATIAL_EXPERIMENT_ID, "spatial_confidence.csv")
DEFAULT_OUTPUT_DIR = experiment_artifact_dir(EXPERIMENT_ID)

SUBSET_DEFINITIONS = {
    "all_available_spatial_confidence": lambda df: pd.Series(True, index=df.index),
    "point_label_share_250m_ge_0.8": lambda df: df["point_label_share_250m"] >= 0.8,
    "point_label_share_500m_ge_0.8": lambda df: df["point_label_share_500m"] >= 0.8,
    "point_label_share_250m_ge_0.9": lambda df: df["point_label_share_250m"] >= 0.9,
    "dominant_matches_point_label_250m": lambda df: df["dominant_matches_point_label_250m"],
    "dominant_matches_point_label_500m": lambda df: df["dominant_matches_point_label_500m"],
}
TEXT_PAIRS = {
    "summary": "summary_shuffled",
    "summary_no_place": "summary_no_place_shuffled",
    "content": "content_shuffled",
    "landuse_evidence_summary": "landuse_evidence_summary_shuffled",
}


def spatial_eval_manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def spatial_eval_summary_path(output_dir: Path) -> Path:
    return output_dir / "summary.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--spatial-confidence-path", type=Path, default=DEFAULT_SPATIAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--parent-experiment-id", default=PARENT_EXPERIMENT_ID)
    parser.add_argument("--spatial-confidence-experiment-id", default=SPATIAL_EXPERIMENT_ID)
    return parser.parse_args(argv)


def _subset_mask(spatial: pd.DataFrame, subset_name: str) -> pd.Series:
    values = SUBSET_DEFINITIONS[subset_name](spatial)
    return parse_boolish_series(values)


def _primary_score(row: dict[str, Any]) -> tuple[str, float]:
    if row["task"] == "corine_level2":
        return "balanced_accuracy", float(row["balanced_accuracy"])
    return "exact_match_accuracy", float(row["exact_match_accuracy"])


def _class_distribution(records: pd.DataFrame, task: str) -> list[dict[str, Any]]:
    labels = (
        CORINE_LEVEL2_DESCRIPTIONS.keys()
        if task == "corine_level2"
        else label_universe(records, task)
    )
    total = len(records)
    rows = []
    for label in sorted(labels):
        if task == "corine_level2":
            support = sum(str(value) == label for value in records["target"])
        else:
            support = sum(label in [str(v) for v in values] for values in records["target"])
        rows.append({"label": label, "support": support, "share": safe_div(support, total)})
    return rows


def evaluate(
    parent_dir: Path,
    spatial_path: Path,
    output_dir: Path,
    *,
    experiment_id: str = EXPERIMENT_ID,
    parent_experiment_id: str = PARENT_EXPERIMENT_ID,
    spatial_confidence_experiment_id: str = SPATIAL_EXPERIMENT_ID,
) -> None:
    spatial = load_spatial_confidence(spatial_path)
    prediction_frames = [load_prediction_records(parent_dir)]
    all_records = pd.concat(prediction_frames, ignore_index=True)
    joined = all_records.merge(spatial, on="pageid", how="left", indicator=True)
    joined_with_spatial = joined[joined["_merge"] == "both"].copy()

    osm = joined[joined["task"] == "osm"]
    osm_coverage = {
        "n_osm_predictions_total": int(len(osm)),
        "n_osm_predictions_with_spatial_confidence": int((osm["_merge"] == "both").sum()),
        "n_osm_predictions_missing_spatial_confidence": int((osm["_merge"] == "left_only").sum()),
    }

    overview_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    majority_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for (task, text_source), group in joined_with_spatial.groupby(
        ["task", "text_source"], sort=True
    ):
        labels = (
            sorted(CORINE_LEVEL2_DESCRIPTIONS)
            if task == "corine_level2"
            else label_universe(group, task)
        )
        for subset_name in SUBSET_DEFINITIONS:
            subset = group[_subset_mask(group, subset_name)]
            subset_rows.append(
                {
                    "task": task,
                    "text_source": text_source,
                    "subset": subset_name,
                    "n_parent_predictions": int(
                        len(
                            joined[
                                (joined["task"] == task) & (joined["text_source"] == text_source)
                            ]
                        )
                    ),
                    "n_with_spatial_confidence": int(len(group)),
                    "n_subset": int(len(subset)),
                }
            )
            if subset.empty:
                continue
            if task == "corine_level2":
                metrics, per_class = compute_single_label_subset_metrics(
                    subset,
                    labels,
                    include_records_without_target=True,
                    include_missing_predictions=False,
                )
                for item in per_class:
                    per_class_rows.append(
                        {"task": task, "text_source": text_source, "subset": subset_name, **item}
                    )
            else:
                metrics = compute_multilabel_subset_metrics(
                    subset,
                    labels,
                    require_list_targets=False,
                    denominator_by_predicted=True,
                    include_missing_predictions_in_derived_multilabel_metrics=False,
                )
            row = {"task": task, "text_source": text_source, "subset": subset_name, **metrics}
            overview_rows.append(row)
            majority_rows.append(
                {
                    key: row.get(key, "")
                    for key in [
                        "task",
                        "text_source",
                        "subset",
                        "majority_accuracy",
                        "majority_balanced_accuracy",
                        "majority_macro_f1",
                        "majority_labelset_exact_match_accuracy",
                        "empty_set_exact_match_accuracy",
                    ]
                }
            )
            for dist in _class_distribution(subset, task):
                distribution_rows.append(
                    {"task": task, "text_source": text_source, "subset": subset_name, **dist}
                )

    overview_by_key = {
        (row["task"], row["text_source"], row["subset"]): row for row in overview_rows
    }
    delta_rows: list[dict[str, Any]] = []
    for task in sorted(set(all_records["task"])):
        for source, shuffled in TEXT_PAIRS.items():
            for subset_name in SUBSET_DEFINITIONS:
                left = overview_by_key.get((task, source, subset_name))
                right = overview_by_key.get((task, shuffled, subset_name))
                if not left or not right:
                    continue
                metric_name, left_score = _primary_score(left)
                _, right_score = _primary_score(right)
                delta_rows.append(
                    {
                        "task": task,
                        "text_source": source,
                        "shuffled_text_source": shuffled,
                        "subset": subset_name,
                        "primary_metric": metric_name,
                        "aligned_score": left_score,
                        "shuffled_score": right_score,
                        "delta": left_score - right_score,
                        "n_aligned": left["n"],
                        "n_shuffled": right["n"],
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("overview_spatial_subsets", overview_rows, "Spatial Subset Overview"),
        ("subset_counts", subset_rows, "Spatial Subset Counts"),
        ("shuffled_delta_spatial_subsets", delta_rows, "Spatial Subset Shuffled Deltas"),
        ("majority_baselines_spatial_subsets", majority_rows, "Spatial Subset Majority Baselines"),
        (
            "per_class_metrics_corine_spatial_subsets",
            per_class_rows,
            "CORINE Per-Class Spatial Metrics",
        ),
        (
            "class_distribution_by_spatial_subset",
            distribution_rows,
            "Class Distribution by Spatial Subset",
        ),
    ]
    for stem, rows, title in outputs:
        if stem == "per_class_metrics_corine_spatial_subsets":
            write_dict_rows_csv_atomic(output_dir / f"{stem}.csv", rows)
        else:
            write_dict_rows_table_pair_atomic(
                output_dir=output_dir,
                stem=stem,
                title=title,
                rows=rows,
            )

    manifest = {
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "spatial_confidence_experiment_id": spatial_confidence_experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_llm_rerun": True,
        "subset_definitions": list(SUBSET_DEFINITIONS),
        "input_paths": {
            "parent_experiment_dir": str(parent_dir),
            "spatial_confidence_path": str(spatial_path),
        },
        "number_of_predictions_loaded": int(len(all_records)),
        "number_of_predictions_evaluated_per_subset": subset_rows,
        "metrics_computed": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "exact_match_accuracy",
            "micro_f1",
            "jaccard",
            "hamming_loss",
        ],
        "osm_spatial_join_coverage": osm_coverage,
    }
    write_json_atomic(spatial_eval_manifest_path(output_dir), manifest, indent=2)
    _write_summary(
        spatial_eval_summary_path(output_dir),
        overview_rows,
        osm_coverage,
        experiment_id=experiment_id,
        parent_experiment_id=parent_experiment_id,
        spatial_confidence_experiment_id=spatial_confidence_experiment_id,
    )


def _beats(row: dict[str, Any], metric: str, baseline: str) -> str:
    value = row.get(metric)
    base = row.get(baseline)
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
        return "n/a"
    return "yes" if value > base else "no"


def _write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    osm_coverage: dict[str, int],
    *,
    experiment_id: str,
    parent_experiment_id: str,
    spatial_confidence_experiment_id: str,
) -> None:
    corine_rows = [row for row in rows if row["task"] == "corine_level2"]
    lines = [
        f"# {experiment_id}",
        "",
        "This experiment reevaluates frozen predictions on CORINE spatial-confidence subsets. No LLM was rerun.",
        "",
        "## Previous Baseline/Shuffled Experiment",
        "",
        f"- parent experiment: `{parent_experiment_id}`",
        f"- spatial-confidence experiment: `{spatial_confidence_experiment_id}`",
        "- parent predictions are treated as frozen inputs and are not modified.",
        "- aligned and shuffled text runs are compared again only after spatial-confidence filtering.",
        "",
        "## Spatial-Confidence Diagnostics",
        "",
        "- `all_available_spatial_confidence` means predictions whose pageid has a CORINE spatial-confidence row.",
        "- high-purity subsets are based on `point_label_share`, the area share of the original point label around the coordinate.",
        "- dominant-match subsets retain points where the dominant surrounding CORINE class matches the point label.",
        "",
        "## OSM Spatial Coverage",
        "",
        f"- total OSM prediction records: {osm_coverage['n_osm_predictions_total']}",
        f"- with CORINE spatial confidence: {osm_coverage['n_osm_predictions_with_spatial_confidence']}",
        f"- missing CORINE spatial confidence: {osm_coverage['n_osm_predictions_missing_spatial_confidence']}",
        "",
        "## Spatial-Subset Reevaluation Results",
        "",
        "CORINE rows below state whether each frozen run beats the recomputed subset-specific majority baseline under accuracy, balanced accuracy, and macro-F1.",
        "",
        "| run | subset | beats majority accuracy | beats majority balanced accuracy | beats majority macro-F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in corine_rows:
        lines.append(
            f"| {row['text_source']} | {row['subset']} | "
            f"{_beats(row, 'accuracy', 'majority_accuracy')} | "
            f"{_beats(row, 'balanced_accuracy', 'majority_balanced_accuracy')} | "
            f"{_beats(row, 'macro_f1', 'majority_macro_f1')} |"
        )
    write_text_atomic(path, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        args.parent_experiment_dir,
        args.spatial_confidence_path,
        args.output_dir,
        experiment_id=args.experiment_id,
        parent_experiment_id=args.parent_experiment_id,
        spatial_confidence_experiment_id=args.spatial_confidence_experiment_id,
    )


if __name__ == "__main__":
    main()
