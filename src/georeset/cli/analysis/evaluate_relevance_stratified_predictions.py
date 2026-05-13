"""Evaluate frozen classification predictions under relevance and spatial filters."""

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

DEFAULT_PARENT_DIRS = [
    Path("data/experiments/article_text_classification_e2e_with_shuffled_control_v1"),
    Path(
        "data/experiments/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0"
    ),
]
DEFAULT_EVIDENCE_METADATA_PATH = Path("data/wiki/article_landuse_evidence_summaries.json")
DEFAULT_SPATIAL_CONFIDENCE_PATH = Path(
    "data/experiments/corine_spatial_confidence_v1/spatial_confidence.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/experiments/article_text_classification_relevance_stratified_v1")
DEFAULT_EXPERIMENT_ID = "article_text_classification_relevance_stratified_v1"

SHUFFLED_TEXT_SOURCE_PAIRS = {
    "summary": "summary_shuffled",
    "summary_no_place": "summary_no_place_shuffled",
    "content": "content_shuffled",
    "landuse_evidence_summary": "landuse_evidence_summary_shuffled",
}

MetricName = Literal["precision", "recall", "f1"]

EVIDENCE_TYPES = [
    "forest",
    "agriculture",
    "vineyard",
    "pasture",
    "water",
    "wetland",
    "shrubland",
    "bare_ground",
    "urban_or_artificial",
    "relief_or_geology",
    "habitat_or_ecology",
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
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=DEFAULT_EXPERIMENT_ID,
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], read_json_file(path))


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


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
    write_markdown_table_atomic(
        path,
        title=title,
        rows=rows,
        columns=sorted({key for row in rows for key in row}) if rows else [],
    )


def _prediction_identity(path: Path) -> tuple[str, str]:
    stem = path.name.removesuffix("_predictions.json")
    if stem.startswith("corine_level2_"):
        return "corine_level2", stem.removeprefix("corine_level2_")
    if stem.startswith("osm_"):
        return "osm", stem.removeprefix("osm_")
    raise ValueError(f"Unknown prediction file name: {path.name}")


def load_prediction_records(experiment_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(experiment_dir.glob("*_predictions.json")):
        task, text_source = _prediction_identity(path)
        predictions = _load_json(path)
        for pageid, payload in predictions.items():
            if not isinstance(payload, dict):
                continue
            row = {
                "pageid": str(payload.get("pageid", pageid)),
                "task": task,
                "text_source": text_source,
                "target": payload.get("target"),
                "prediction": payload.get("prediction"),
                "parse_status": payload.get("parse_status"),
                "metadata": payload.get("metadata", {}),
            }
            target = payload.get("target")
            if isinstance(target, list):
                row["target"] = [str(v) for v in target]
            elif target is not None:
                row["target"] = str(target)
            rows.append(row)
    return pd.DataFrame(rows)


def load_evidence_metadata(path: Path) -> pd.DataFrame:
    raw = _load_json(path)
    records: list[dict[str, Any]] = []
    for pageid_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        pageid = payload.get("pageid", pageid_key)
        if pageid is None:
            continue
        evidence_types = payload.get("evidence_types", [])
        if not isinstance(evidence_types, list):
            evidence_types = []
        records.append(
            {
                "pageid": str(pageid),
                "landcover_relevance": payload.get("landcover_relevance"),
                "uncertainty": payload.get("uncertainty"),
                "evidence_types": [str(value) for value in evidence_types],
                "evidence_sentences_count": int(payload.get("evidence_sentences_count", 0) or 0),
            }
        )
    return pd.DataFrame(records)


def load_spatial_confidence(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"pageid": str})
    df["pageid"] = df["pageid"].astype(str)
    return df


def join_metadata_with_evidence(
    records: pd.DataFrame, evidence_metadata: pd.DataFrame
) -> pd.DataFrame:
    if evidence_metadata.empty:
        return records.assign(
            landcover_relevance=None,
            uncertainty=None,
            evidence_types=[[] for _ in range(len(records))],
            evidence_sentences_count=0,
            evidence_metadata_present=False,
        )
    joined = records.merge(evidence_metadata, on="pageid", how="left")
    joined["evidence_metadata_present"] = joined["landcover_relevance"].notna()
    joined["evidence_types"] = joined["evidence_types"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    joined["evidence_sentences_count"] = joined["evidence_sentences_count"].fillna(0).astype(int)
    return joined


def define_relevance_subsets(records: pd.DataFrame) -> dict[str, pd.Series]:
    relevance = records.get("landcover_relevance", pd.Series([""] * len(records), index=records.index)).astype(
        "string"
    ).str.lower()
    uncertainty = records.get("uncertainty", pd.Series([""] * len(records), index=records.index)).astype(
        "string"
    ).str.lower()
    sentence_count = pd.to_numeric(
        records.get("evidence_sentences_count", pd.Series([0] * len(records), index=records.index)),
        errors="coerce",
    ).fillna(0)
    spatial_share = pd.to_numeric(
        records.get("point_label_share_250m", pd.Series([0.0] * len(records), index=records.index)),
        errors="coerce",
    )
    return {
        "all": pd.Series(True, index=records.index),
        "relevance_none": relevance == "none",
        "relevance_low": relevance == "low",
        "relevance_medium": relevance == "medium",
        "relevance_high": relevance == "high",
        "relevance_low_medium_high": relevance.isin(["low", "medium", "high"]),
        "relevance_medium_high": relevance.isin(["medium", "high"]),
        "evidence_sentences_count_ge_1": sentence_count >= 1,
        "evidence_sentences_count_ge_2": sentence_count >= 2,
        "uncertainty_low": uncertainty == "low",
        "uncertainty_low_medium": uncertainty.isin(["low", "medium"]),
        "point_label_share_250m_ge_0.8": spatial_share >= 0.8,
        "point_label_share_250m_ge_0.8_and_relevance_medium_high": (relevance.isin(["medium", "high"]))
        & (spatial_share >= 0.8),
        "point_label_share_250m_ge_0.8_and_evidence_sentences_count_ge_1": (sentence_count >= 1)
        & (spatial_share >= 0.8),
    }


def define_relevance_and_spatial_subsets(records: pd.DataFrame) -> dict[str, pd.Series]:
    relevance = define_relevance_subsets(records)
    spatial = (pd.to_numeric(records["point_label_share_250m"], errors="coerce") >= 0.8).fillna(False)
    return {
        f"{name}_and_point_label_share_250m_ge_0.8": subset & spatial
        for name, subset in relevance.items()
        if name != "all"
        and not name.startswith("point_label_share_250m_ge_0.8")
    } | {"all_and_point_label_share_250m_ge_0.8": spatial}


def define_evidence_type_subsets(records: pd.DataFrame) -> dict[str, pd.Series]:
    evidence_lists = records["evidence_types"]
    def _contains_evidence_type(values: object, evidence_type: str) -> bool:
        return evidence_type in (values if isinstance(values, list) else [])

    return {
        f"evidence_type_{name}": evidence_lists.apply(_contains_evidence_type, evidence_type=name)
        for name in EVIDENCE_TYPES
    }


def _weighted_from_per_label(
    per_label: dict[str, PerLabelMetric], metric: MetricName
) -> float:
    total_support = sum(values["support"] for values in per_label.values())
    if total_support == 0:
        return 0.0
    return _safe_div(
        sum(values[metric] * values["support"] for values in per_label.values()),
        total_support,
    )


def _single_label_metrics(
    records: pd.DataFrame, labels: list[str]
) -> tuple[SpatialSubsetMetricResult, list[dict[str, Any]]]:
    y_true = {
        row["pageid"]: str(row["target"]) for row in records.to_dict("records") if row["target"] is not None
    }
    ok = records[records["parse_status"] == "ok"]
    y_pred = {
        row["pageid"]: str(row["prediction"])
        for row in ok.to_dict("records")
        if row["target"] is not None and row["prediction"] is not None
    }
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
        "balanced_accuracy": base_metrics["macro_recall"],
    }
    metrics["weighted_precision"] = _weighted_from_per_label(base_metrics["per_label"], "precision")
    metrics["weighted_recall"] = _weighted_from_per_label(base_metrics["per_label"], "recall")
    metrics["weighted_f1"] = _weighted_from_per_label(base_metrics["per_label"], "f1")
    if y_true:
        majority_label = Counter(y_true.values()).most_common(1)[0][0]
        majority_pred = dict.fromkeys(y_true, majority_label)
    else:
        majority_pred = {}
    majority = single_label_metrics(y_true, majority_pred, labels)
    metrics["majority_accuracy"] = majority["accuracy"]
    metrics["majority_balanced_accuracy"] = majority["macro_recall"]
    metrics["majority_macro_f1"] = majority["macro_f1"]
    metrics["delta_vs_majority_accuracy"] = metrics["accuracy"] - metrics["majority_accuracy"]
    metrics["delta_vs_majority_balanced_accuracy"] = (
        metrics["balanced_accuracy"] - metrics["majority_balanced_accuracy"]
    )
    metrics["delta_vs_majority_macro_f1"] = metrics["macro_f1"] - metrics["majority_macro_f1"]
    per_class_rows = [
        {
            "label": label,
            "support": values["support"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
        }
        for label, values in base_metrics["per_label"].items()
    ]
    return metrics, per_class_rows


def _multi_label_metrics(
    records: pd.DataFrame, labels: list[str]
) -> SpatialSubsetMetricResult:
    rows = records.to_dict("records")
    y_true = {row["pageid"]: row["target"] for row in rows if isinstance(row["target"], list)}
    ok = records[records["parse_status"] == "ok"]
    y_pred = {
        row["pageid"]: (
            [str(value) for value in row["prediction"]] if isinstance(row["prediction"], list) else []
        )
        for row in ok.to_dict("records")
        if row["pageid"] in y_true
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
    jaccards: list[float] = []
    hamming_errors = 0
    for pageid, true_values in y_true.items():
        pred_values = y_pred.get(pageid, [])
        true_set = set(true_values)
        pred_set = set(pred_values)
        union = true_set | pred_set
        if union:
            jaccards.append(_safe_div(len(true_set & pred_set), len(union)))
        else:
            jaccards.append(1.0)
        hamming_errors += len(true_set ^ pred_set)
    metrics["jaccard"] = _safe_div(sum(jaccards), len(jaccards))
    total_labels = len(labels) if labels else 1
    metrics["hamming_loss"] = _safe_div(hamming_errors, max(len(y_true) * total_labels, 1))
    target_keys = [json.dumps(sorted(values), ensure_ascii=False) for values in y_true.values()]
    majority_key = Counter(target_keys).most_common(1)[0][0] if target_keys else "[]"
    majority_set = json.loads(majority_key)
    majority_pred = {pageid: list(majority_set) for pageid in y_true}
    empty_pred: dict[str, list[str]] = {pageid: [] for pageid in y_true}
    metrics["majority_labelset_exact_match_accuracy"] = multilabel_metrics(y_true, majority_pred, labels)[
        "exact_match_accuracy"
    ]
    metrics["empty_set_exact_match_accuracy"] = multilabel_metrics(y_true, empty_pred, labels)[
        "exact_match_accuracy"
    ]
    return metrics


def _infer_model(records: pd.DataFrame, parent_dir: Path) -> str:
    for metadata in records["metadata"]:
        if isinstance(metadata, dict) and isinstance(metadata.get("model"), str):
            return str(metadata["model"])
    if "__gemma4_31b_it_q4_0" in str(parent_dir):
        return "gemma-4-31B-it-Q4_0.gguf"
    return "Qwen3.6-27B-Q4_0.gguf"


def _metric_name(task: str) -> str:
    return "balanced_accuracy" if task == "corine_level2" else "exact_match_accuracy"


def _compute_shuffled_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["model"], row["task"], row["text_source"], row["relevance_subset"]): row for row in rows
    }
    deltas: list[dict[str, Any]] = []
    for row in rows:
        metric = _metric_name(row["task"])
        source = row["text_source"]
        shuffled = SHUFFLED_TEXT_SOURCE_PAIRS.get(source)
        if not shuffled:
            continue
        ref = lookup.get((row["model"], row["task"], shuffled, row["relevance_subset"]))
        if not ref:
            continue
        delta: float | str = ""
        if isinstance(row.get(metric), (int, float)) and isinstance(ref.get(metric), (int, float)):
            delta = float(row[metric]) - float(ref[metric])
        deltas.append(
            {
                "model": row["model"],
                "task": row["task"],
                "text_source": source,
                "shuffled_text_source": shuffled,
                "relevance_subset": row["relevance_subset"],
                "primary_metric": metric,
                "aligned_score": row.get(metric, ""),
                "shuffled_score": ref.get(metric, ""),
                "delta": delta,
                "n_aligned": row.get("n", ""),
                "n_shuffled": ref.get("n", ""),
            }
        )
    return deltas


def _compute_model_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["text_source"], row["relevance_subset"]), {})[
            row["model"]
        ] = row
    output = []
    for (task, text_source, subset), model_rows in grouped.items():
        models = sorted(model_rows)
        if len(models) < 2:
            continue
        model_a = models[0]
        model_b = models[1]
        row_a = model_rows[model_a]
        row_b = model_rows[model_b]
        metric = _metric_name(task)
        score_a = row_a.get(metric)
        score_b = row_b.get(metric)
        if not isinstance(score_a, (int, float)) or not isinstance(score_b, (int, float)):
            continue
        output.append(
            {
                "task": task,
                "text_source": text_source,
                "relevance_subset": subset,
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "score_a": score_a,
                "score_b": score_b,
                "delta_a_minus_b": float(score_a) - float(score_b),
                "n_a": row_a.get("n"),
                "n_b": row_b.get("n"),
            }
        )
    return output


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model", "")),
        str(row.get("task", "")),
        str(row.get("text_source", "")),
        str(row.get("relevance_subset", row.get("evidence_type_subset", ""))),
    )


def evaluate(
    parent_dirs: list[Path],
    evidence_metadata_path: Path,
    spatial_confidence_path: Path,
    output_dir: Path,
    *,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    evidence = load_evidence_metadata(evidence_metadata_path)
    spatial = load_spatial_confidence(spatial_confidence_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[pd.DataFrame] = []
    for parent_dir in parent_dirs:
        records = load_prediction_records(parent_dir)
        if records.empty:
            continue
        records = join_metadata_with_evidence(records, evidence)
        records = records.merge(spatial, on="pageid", how="left")
        records["model"] = _infer_model(records, parent_dir)
        records["source_parent_experiment_dir"] = str(parent_dir)
        all_records.append(records)

    if not all_records:
        raise ValueError("No prediction records loaded from input experiments.")

    merged = pd.concat(all_records, ignore_index=True)
    overview_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    overview_spatial_rows: list[dict[str, Any]] = []
    majority_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for parent_dir, parent_records in merged.groupby("source_parent_experiment_dir"):
        model = str(parent_records["model"].iloc[0])
        for task, text_source in parent_records.groupby(["task", "text_source"]).groups:
            group = parent_records[parent_records["task"].eq(task) & parent_records["text_source"].eq(text_source)]
            task_records = group.to_dict("records")
            if task == "corine_level2":
                labels = sorted(set(CORINE_LEVEL2_DESCRIPTIONS))
            else:
                labels = sorted(
                    {
                        str(label)
                        for record in task_records
                        for label in (
                            record["target"] if isinstance(record["target"], list) else [record["target"]]
                        )
                        if record["target"] is not None and label is not None
                    }
                )

            total = len(group)
            total_ok = len(group[group["parse_status"] == "ok"])
            total_missing_metadata = int((~group["evidence_metadata_present"]).sum())
            distribution_rows.append(
                {
                    "model": model,
                    "source_parent_experiment_dir": parent_dir,
                    "task": task,
                    "text_source": text_source,
                    "dimension": "total_predictions",
                    "value": "total",
                    "count": total,
                    "n_predicted_ok": total_ok,
                    "n_parse_error": total - total_ok,
                }
            )
            distribution_rows.append(
                {
                    "model": model,
                    "source_parent_experiment_dir": parent_dir,
                    "task": task,
                    "text_source": text_source,
                    "dimension": "missing_evidence_records",
                    "value": "missing_metadata",
                    "count": total_missing_metadata,
                    "n_predicted_ok": total_ok,
                    "n_parse_error": total - total_ok,
                }
            )
            for field in ("landcover_relevance", "uncertainty"):
                for value, count in (
                    group[field].fillna("missing").astype(str).value_counts(dropna=False).items()
                ):
                    distribution_rows.append(
                        {
                            "model": model,
                            "source_parent_experiment_dir": parent_dir,
                            "task": task,
                            "text_source": text_source,
                            "dimension": field,
                            "value": value,
                            "count": int(count),
                            "n_predicted_ok": total_ok,
                            "n_parse_error": total - total_ok,
                        }
                    )

            for subset_name, mask in define_relevance_subsets(group).items():
                subset = group[mask]
                if task == "corine_level2":
                    metrics, per_class = _single_label_metrics(subset, labels)
                else:
                    metrics = _multi_label_metrics(subset, labels)
                    per_class = []
                row: dict[str, Any] = {
                    "model": model,
                    "task": task,
                    "text_source": text_source,
                    "source_parent_experiment_dir": parent_dir,
                    "relevance_subset": subset_name,
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
                    "majority_balanced_accuracy": metrics.get(
                        "majority_balanced_accuracy", ""
                    ),
                    "majority_macro_f1": metrics.get("majority_macro_f1", ""),
                    "majority_labelset_exact_match_accuracy": metrics.get(
                        "majority_labelset_exact_match_accuracy", ""
                    ),
                    "empty_set_exact_match_accuracy": metrics.get("empty_set_exact_match_accuracy", ""),
                    "exact_match_accuracy": metrics.get("exact_match_accuracy", ""),
                }
                overview_rows.append(row)
                majority_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "text_source": text_source,
                        "source_parent_experiment_dir": parent_dir,
                        "relevance_subset": subset_name,
                        "n": metrics["n"],
                        "n_predicted_ok": metrics["n_predicted_ok"],
                        "n_parse_error": metrics["n_parse_error"],
                        "coverage": metrics["coverage"],
                        "majority_accuracy": metrics.get("majority_accuracy", ""),
                        "majority_balanced_accuracy": metrics.get("majority_balanced_accuracy", ""),
                        "majority_macro_f1": metrics.get("majority_macro_f1", ""),
                        "majority_labelset_exact_match_accuracy": metrics.get(
                            "majority_labelset_exact_match_accuracy", ""
                        ),
                        "empty_set_exact_match_accuracy": metrics.get("empty_set_exact_match_accuracy", ""),
                    }
                )
                for item in per_class:
                    per_class_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "text_source": text_source,
                            "source_parent_experiment_dir": parent_dir,
                            "relevance_subset": subset_name,
                            **item,
                        }
                    )

            for evidence_name, mask in define_evidence_type_subsets(group).items():
                subset = group[mask]
                if task == "corine_level2":
                    metrics, _ = _single_label_metrics(subset, labels)
                else:
                    metrics = _multi_label_metrics(subset, labels)
                evidence_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "text_source": text_source,
                        "source_parent_experiment_dir": parent_dir,
                        "evidence_type_subset": evidence_name,
                        "n": metrics["n"],
                        "n_predicted_ok": metrics["n_predicted_ok"],
                        "n_parse_error": metrics["n_parse_error"],
                        "coverage": metrics["coverage"],
                        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
                        "macro_f1": metrics.get("macro_f1", ""),
                        "exact_match_accuracy": metrics.get("exact_match_accuracy", ""),
                        "micro_f1": metrics.get("micro_f1", ""),
                        "macro_recall": metrics.get("macro_recall", ""),
                        "jaccard": metrics.get("jaccard", ""),
                        "hamming_loss": metrics.get("hamming_loss", ""),
                    }
                )

            for subset_name, mask in define_relevance_and_spatial_subsets(group).items():
                subset = group[mask]
                if task == "corine_level2":
                    metrics, _ = _single_label_metrics(subset, labels)
                else:
                    metrics = _multi_label_metrics(subset, labels)
                overview_spatial_rows.append(
                    {
                        "model": model,
                        "task": task,
                        "text_source": text_source,
                        "source_parent_experiment_dir": parent_dir,
                        "relevance_subset": subset_name,
                        "n": metrics["n"],
                        "n_predicted_ok": metrics["n_predicted_ok"],
                        "n_parse_error": metrics["n_parse_error"],
                        "coverage": metrics["coverage"],
                        "accuracy": metrics.get("accuracy", ""),
                        "balanced_accuracy": metrics.get("balanced_accuracy", ""),
                        "macro_f1": metrics.get("macro_f1", ""),
                        "exact_match_accuracy": metrics.get("exact_match_accuracy", ""),
                        "micro_f1": metrics.get("micro_f1", ""),
                    }
                )

    shuffled_rows = _compute_shuffled_deltas(overview_rows)
    comparison_rows = _compute_model_comparison(overview_rows)

    _write_csv(output_dir / "overview_by_relevance.csv", sorted(overview_rows, key=_row_sort_key))
    _write_csv(output_dir / "overview_by_relevance_and_spatial_confidence.csv", sorted(overview_spatial_rows, key=_row_sort_key))
    _write_csv(output_dir / "overview_by_evidence_type.csv", sorted(evidence_rows, key=_row_sort_key))
    _write_csv(output_dir / "shuffled_delta_by_relevance.csv", sorted(shuffled_rows, key=_row_sort_key))
    _write_csv(output_dir / "majority_baselines_by_relevance.csv", sorted(majority_rows, key=_row_sort_key))
    _write_csv(output_dir / "per_class_corine_by_relevance.csv", sorted(per_class_rows, key=_row_sort_key))
    _write_csv(output_dir / "model_comparison_by_relevance.csv", sorted(comparison_rows, key=_row_sort_key))
    _write_csv(output_dir / "evidence_metadata_distribution.csv", sorted(distribution_rows, key=_row_sort_key))

    _write_md(output_dir / "overview_by_relevance.md", "Relevance Subset Metrics", sorted(overview_rows, key=_row_sort_key))
    _write_md(
        output_dir / "overview_by_relevance_and_spatial_confidence.md",
        "Relevance + Spatial Confidence Metrics",
        sorted(overview_spatial_rows, key=_row_sort_key),
    )
    _write_md(output_dir / "overview_by_evidence_type.md", "Evidence-Type Metrics", sorted(evidence_rows, key=_row_sort_key))
    _write_md(output_dir / "shuffled_delta_by_relevance.md", "Shuffled Deltas by Relevance", sorted(shuffled_rows, key=_row_sort_key))
    _write_md(output_dir / "majority_baselines_by_relevance.md", "Majority Baselines by Relevance", sorted(majority_rows, key=_row_sort_key))
    _write_md(output_dir / "per_class_corine_by_relevance.md", "Per-Class CORINE Metrics", sorted(per_class_rows, key=_row_sort_key))
    _write_md(output_dir / "model_comparison_by_relevance.md", "Model Comparison by Relevance", sorted(comparison_rows, key=_row_sort_key))
    _write_md(output_dir / "evidence_metadata_distribution.md", "Evidence Metadata Distribution", sorted(distribution_rows, key=_row_sort_key))

    _write_summary(output_dir, overview_rows, shuffled_rows, comparison_rows)
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "input_paths": {
                "evidence_metadata_path": str(evidence_metadata_path),
                "spatial_confidence_path": str(spatial_confidence_path),
            },
            "source_experiments": sorted(set(merged["source_parent_experiment_dir"])),
            "source_parent_experiment_dirs": sorted(set(merged["source_parent_experiment_dir"])),
            "relevance_subset_definitions": list(define_relevance_subsets(merged).keys()),
            "evidence_type_subset_definitions": [f"evidence_type_{name}" for name in EVIDENCE_TYPES],
            "models": sorted(set(merged["model"])),
            "source_artifacts": {
                "text_sources": sorted(set(merged["text_source"])),
                "tasks": sorted(set(merged["task"])),
                "num_records_loaded": int(len(merged)),
            },
            "metrics_computed": [
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "exact_match_accuracy",
                "jaccard",
                "hamming_loss",
            ],
            "no_llm_rerun": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def _write_summary(
    output_dir: Path,
    overview_rows: list[dict[str, Any]],
    shuffled_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    if not overview_rows:
        write_text_atomic(output_dir / "summary.md", "# article_text_classification_relevance_stratified_v1\n\nNo rows.\n")
        return
    rows = sorted(
        [row for row in comparison_rows if isinstance(row.get("delta_a_minus_b"), (int, float))],
        key=lambda row: abs(row["delta_a_minus_b"]),
        reverse=True,
    )
    best_model_comparison = rows[0] if rows else None
    shuffled_rows_sorted = sorted(
        [row for row in shuffled_rows if isinstance(row.get("delta"), (int, float))],
        key=lambda row: abs(row["delta"]),
        reverse=True,
    )
    best_shuffled_delta = shuffled_rows_sorted[0] if shuffled_rows_sorted else None

    lines = [
        "# article_text_classification_relevance_stratified_v1",
        "",
        "This experiment reevaluates frozen classification predictions by:",
        "- relevance metadata subsets (`low`, `medium`, `high`, etc.)",
        "- evidence-type membership subsets",
        "- optional spatial-confidence intersections",
        "- task-specific shuffled-control deltas per subset",
        "",
        f"- CORINE rows: {sum(1 for row in overview_rows if row['task']=='corine_level2')}",
        f"- OSM rows: {sum(1 for row in overview_rows if row['task']=='osm')}",
    ]
    if best_model_comparison:
        lines.extend(
            [
                "",
                "## Model Comparison (top delta)",
                f"- {best_model_comparison['model_a']} vs {best_model_comparison['model_b']} "
                f"{best_model_comparison['task']} {best_model_comparison['text_source']} "
                f"{best_model_comparison['relevance_subset']}: "
                f"{best_model_comparison['delta_a_minus_b']:.4f}",
            ]
        )
    if best_shuffled_delta:
        lines.extend(
            [
                "",
                "## Shuffled Delta (best aligned advantage)",
                f"- {best_shuffled_delta['model']} {best_shuffled_delta['task']} "
                f"{best_shuffled_delta['text_source']} "
                f"{best_shuffled_delta['relevance_subset']}: "
                f"{best_shuffled_delta['delta']}",
            ]
        )
    write_text_atomic(output_dir / "summary.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(
        parent_dirs=args.parent_experiment_dir or DEFAULT_PARENT_DIRS,
        evidence_metadata_path=args.evidence_metadata_path,
        spatial_confidence_path=args.spatial_confidence_path,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
