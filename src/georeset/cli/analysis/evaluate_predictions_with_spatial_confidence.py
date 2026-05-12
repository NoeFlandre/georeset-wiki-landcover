"""Reevaluate frozen classification predictions on CORINE spatial-confidence subsets."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from georeset.classification.labels import CORINE_LEVEL2_DESCRIPTIONS
from georeset.classification.metrics import multilabel_metrics, single_label_metrics
from georeset.contracts import MultiLabelMetricResult, PerLabelMetric, SpatialSubsetMetricResult
from georeset.utils.json_io import (
    read_json_file,
    write_json_atomic,
    write_markdown_table_atomic,
    write_text_atomic,
)

EXPERIMENT_ID = "article_text_classification_spatial_confidence_v1"
PARENT_EXPERIMENT_ID = "article_text_classification_e2e_with_shuffled_control_v1"
SPATIAL_EXPERIMENT_ID = "corine_spatial_confidence_v1"
DEFAULT_PARENT_DIR = Path(
    "data/experiments/article_text_classification_e2e_with_shuffled_control_v1"
)
DEFAULT_SPATIAL_PATH = Path("data/experiments/corine_spatial_confidence_v1/spatial_confidence.csv")
DEFAULT_OUTPUT_DIR = Path("data/experiments/article_text_classification_spatial_confidence_v1")

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--spatial-confidence-path", type=Path, default=DEFAULT_SPATIAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], read_json_file(path))


def _prediction_identity(path: Path) -> tuple[str, str]:
    stem = path.name.removesuffix("_predictions.json")
    if stem.startswith("corine_level2_"):
        return "corine_level2", stem.removeprefix("corine_level2_")
    if stem.startswith("osm_"):
        return "osm", stem.removeprefix("osm_")
    raise ValueError(f"Unknown prediction file name: {path.name}")


def _records_frame(path: Path) -> pd.DataFrame:
    task, text_source = _prediction_identity(path)
    rows = []
    for pageid, record in _load_json(path).items():
        rows.append(
            {
                "pageid": str(pageid),
                "task": task,
                "text_source": text_source,
                "target": record.get("target"),
                "prediction": record.get("prediction"),
                "parse_status": record.get("parse_status"),
            }
        )
    return pd.DataFrame(rows)


def load_spatial_confidence(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"pageid": str, "point_label": str})
    df["pageid"] = df["pageid"].astype(str)
    for column in df.columns:
        if column.startswith("dominant_matches_point_label_") and df[column].dtype == object:
            df[column] = df[column].map(lambda value: str(value).lower() == "true")
    return df


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _weighted_from_per_label(
    per_label: dict[str, PerLabelMetric], key: Literal["precision", "recall", "f1"]
) -> float:
    total_support = sum(values["support"] for values in per_label.values())
    return _safe_div(
        sum(values[key] * values["support"] for values in per_label.values()),
        total_support,
    )


def _single_metrics(
    records: pd.DataFrame, labels: list[str]
) -> tuple[SpatialSubsetMetricResult, list[dict[str, Any]]]:
    y_true = dict(zip(records["pageid"], records["target"].astype(str), strict=False))
    ok_records = records[records["parse_status"] == "ok"]
    y_pred = dict(zip(ok_records["pageid"], ok_records["prediction"].astype(str), strict=False))
    base_metrics = single_label_metrics(y_true, y_pred, labels)
    metrics: SpatialSubsetMetricResult = {
        "n": base_metrics["n_eligible"],
        "n_predicted_ok": base_metrics["n_predicted_ok"],
        "n_parse_error": base_metrics["n_parse_error"],
        "coverage": base_metrics["coverage"],
        "accuracy": base_metrics["accuracy"],
        "macro_precision": base_metrics["macro_precision"],
        "macro_recall": base_metrics["macro_recall"],
        "macro_f1": base_metrics["macro_f1"],
    }
    metrics["balanced_accuracy"] = metrics["macro_recall"]
    metrics["weighted_precision"] = _weighted_from_per_label(base_metrics["per_label"], "precision")
    metrics["weighted_recall"] = _weighted_from_per_label(base_metrics["per_label"], "recall")
    metrics["weighted_f1"] = _weighted_from_per_label(base_metrics["per_label"], "f1")

    majority_label = Counter(y_true.values()).most_common(1)[0][0] if y_true else None
    majority_pred = dict.fromkeys(y_true, majority_label) if majority_label else {}
    majority = single_label_metrics(y_true, majority_pred, labels)
    metrics["majority_accuracy"] = majority["accuracy"]
    metrics["majority_balanced_accuracy"] = majority["macro_recall"]
    metrics["majority_macro_f1"] = majority["macro_f1"]
    metrics["delta_vs_majority_accuracy"] = metrics["accuracy"] - majority["accuracy"]
    metrics["delta_vs_majority_balanced_accuracy"] = (
        metrics["balanced_accuracy"] - majority["macro_recall"]
    )
    metrics["delta_vs_majority_macro_f1"] = metrics["macro_f1"] - majority["macro_f1"]

    per_class = [
        {
            "label": label,
            "support": values["support"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
        }
        for label, values in base_metrics["per_label"].items()
    ]
    return metrics, per_class


def _label_universe(records: pd.DataFrame) -> list[str]:
    labels: set[str] = set()
    for column in ["target", "prediction"]:
        for values in records[column]:
            if isinstance(values, list):
                labels.update(str(value) for value in values)
    return sorted(labels)


def _multilabel_metrics(records: pd.DataFrame, labels: list[str]) -> SpatialSubsetMetricResult:
    y_true = {row.pageid: [str(v) for v in row.target] for row in records.itertuples()}
    ok_records = records[records["parse_status"] == "ok"]
    y_pred = {
        row.pageid: [str(v) for v in row.prediction] if isinstance(row.prediction, list) else []
        for row in ok_records.itertuples()
    }
    base_metrics: MultiLabelMetricResult = multilabel_metrics(y_true, y_pred, labels)
    metrics: SpatialSubsetMetricResult = {
        "n": base_metrics["n_eligible"],
        "n_predicted_ok": base_metrics["n_predicted_ok"],
        "n_parse_error": base_metrics["n_parse_error"],
        "coverage": base_metrics["coverage"],
        "exact_match_accuracy": base_metrics["exact_match_accuracy"],
        "micro_precision": base_metrics["micro_precision"],
        "micro_recall": base_metrics["micro_recall"],
        "micro_f1": base_metrics["micro_f1"],
        "macro_precision": base_metrics["macro_precision"],
        "macro_recall": base_metrics["macro_recall"],
        "macro_f1": base_metrics["macro_f1"],
    }
    jaccards = []
    hamming_errors = 0
    for pageid, true_values in y_true.items():
        true_set = set(true_values)
        if pageid not in y_pred:
            continue
        pred_set = set(y_pred[pageid])
        union = true_set | pred_set
        jaccards.append(1.0 if not union else len(true_set & pred_set) / len(union))
        hamming_errors += len(true_set ^ pred_set)
    metrics["jaccard"] = _safe_div(sum(jaccards), len(jaccards))
    metrics["hamming_loss"] = _safe_div(hamming_errors, len(y_pred) * len(labels))

    target_keys = [json.dumps(sorted(values), ensure_ascii=False) for values in y_true.values()]
    majority_key = Counter(target_keys).most_common(1)[0][0] if target_keys else "[]"
    majority_set = json.loads(majority_key)
    majority_pred = {pageid: list(majority_set) for pageid in y_true}
    empty_pred: dict[str, list[str]] = {pageid: [] for pageid in y_true}
    majority = multilabel_metrics(y_true, majority_pred, labels)
    empty = multilabel_metrics(y_true, empty_pred, labels)
    metrics["majority_labelset_exact_match_accuracy"] = majority["exact_match_accuracy"]
    metrics["empty_set_exact_match_accuracy"] = empty["exact_match_accuracy"]
    return metrics


def _subset_mask(spatial: pd.DataFrame, subset_name: str) -> pd.Series:
    return SUBSET_DEFINITIONS[subset_name](spatial).fillna(False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_text_atomic(path, "")
        return
    fieldnames = sorted({key for row in rows for key in row})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, output.getvalue())


def _write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    write_markdown_table_atomic(path, title=title, rows=rows)


def _primary_score(row: dict[str, Any]) -> tuple[str, float]:
    if row["task"] == "corine_level2":
        return "balanced_accuracy", float(row["balanced_accuracy"])
    return "exact_match_accuracy", float(row["exact_match_accuracy"])


def _class_distribution(records: pd.DataFrame, task: str) -> list[dict[str, Any]]:
    labels = (
        CORINE_LEVEL2_DESCRIPTIONS.keys() if task == "corine_level2" else _label_universe(records)
    )
    total = len(records)
    rows = []
    for label in sorted(labels):
        if task == "corine_level2":
            support = sum(str(value) == label for value in records["target"])
        else:
            support = sum(label in [str(v) for v in values] for values in records["target"])
        rows.append({"label": label, "support": support, "share": _safe_div(support, total)})
    return rows


def evaluate(parent_dir: Path, spatial_path: Path, output_dir: Path) -> None:
    spatial = load_spatial_confidence(spatial_path)
    prediction_frames = [
        _records_frame(path) for path in sorted(parent_dir.glob("*_predictions.json"))
    ]
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
            else _label_universe(group)
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
                metrics, per_class = _single_metrics(subset, labels)
                for item in per_class:
                    per_class_rows.append(
                        {"task": task, "text_source": text_source, "subset": subset_name, **item}
                    )
            else:
                metrics = _multilabel_metrics(subset, labels)
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
        _write_csv(output_dir / f"{stem}.csv", rows)
        if stem != "per_class_metrics_corine_spatial_subsets":
            _write_md(output_dir / f"{stem}.md", title, rows)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "spatial_confidence_experiment_id": SPATIAL_EXPERIMENT_ID,
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
    write_json_atomic(output_dir / "manifest.json", manifest, indent=2)
    _write_summary(output_dir / "summary.md", overview_rows, osm_coverage)


def _beats(row: dict[str, Any], metric: str, baseline: str) -> str:
    value = row.get(metric)
    base = row.get(baseline)
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
        return "n/a"
    return "yes" if value > base else "no"


def _write_summary(path: Path, rows: list[dict[str, Any]], osm_coverage: dict[str, int]) -> None:
    corine_rows = [row for row in rows if row["task"] == "corine_level2"]
    lines = [
        "# Article-Text Classification Spatial Confidence v1",
        "",
        "This experiment reevaluates frozen predictions on CORINE spatial-confidence subsets. No LLM was rerun.",
        "",
        "## Previous Baseline/Shuffled Experiment",
        "",
        f"- parent experiment: `{PARENT_EXPERIMENT_ID}`",
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
    evaluate(args.parent_experiment_dir, args.spatial_confidence_path, args.output_dir)


if __name__ == "__main__":
    main()
