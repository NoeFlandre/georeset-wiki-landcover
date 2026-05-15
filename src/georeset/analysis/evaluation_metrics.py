"""Shared metric helpers for CLI analysis scripts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal, cast

import pandas as pd

from georeset.classification.metrics import multilabel_metrics, single_label_metrics
from georeset.contracts import PerLabelMetric, SpatialSubsetMetricResult

MetricName = Literal["precision", "recall", "f1"]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


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


def _as_records_frame(records: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], records.to_dict("records"))


def compute_single_label_subset_metrics(
    records: pd.DataFrame,
    labels: list[str],
    *,
    include_records_without_target: bool = True,
    include_missing_predictions: bool = True,
) -> tuple[SpatialSubsetMetricResult, list[dict[str, Any]]]:
    rows = _as_records_frame(records)
    if include_records_without_target:
        y_true = {str(row["pageid"]): str(row["target"]) for row in rows}
    else:
        y_true = {
            str(row["pageid"]): str(row["target"]) for row in rows if row["target"] is not None
        }

    ok_records = records[records["parse_status"] == "ok"]
    ok_rows = _as_records_frame(ok_records)
    if include_missing_predictions:
        y_pred = {
            str(row["pageid"]): str(row["prediction"])
            for row in ok_rows
            if str(row["pageid"]) in y_true
        }
    else:
        y_pred = {
            str(row["pageid"]): str(row["prediction"])
            for row in ok_rows
            if row["prediction"] is not None and str(row["pageid"]) in y_true
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
    }
    metrics["balanced_accuracy"] = metrics["macro_recall"]
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


def _labels_for_record(record_value: Any) -> Iterable[Any]:
    if isinstance(record_value, list):
        return record_value
    if isinstance(record_value, tuple):
        return record_value
    if record_value is None:
        return []
    # Match existing behavior in CLI scripts where non-list CORINE targets are treated
    # as iterables in direct list-comprehension style.
    return cast(Iterable[Any], record_value)


def compute_multilabel_subset_metrics(
    records: pd.DataFrame,
    labels: list[str],
    *,
    require_list_targets: bool = False,
    denominator_by_predicted: bool = True,
    include_missing_predictions_in_derived_multilabel_metrics: bool = True,
) -> SpatialSubsetMetricResult:
    rows = _as_records_frame(records)
    if require_list_targets:
        y_true = {
            str(row["pageid"]): [str(value) for value in row["target"]]
            for row in rows
            if isinstance(row["target"], list)
        }
    else:
        y_true = {
            str(row["pageid"]): [str(value) for value in _labels_for_record(row["target"])]
            for row in rows
        }
    ok_rows = _as_records_frame(records[records["parse_status"] == "ok"])
    y_pred = {
        str(row["pageid"]): [str(value) for value in row["prediction"]]
        if isinstance(row["prediction"], list)
        else []
        for row in ok_rows
        if str(row["pageid"]) in y_true
    }
    base_metrics = multilabel_metrics(y_true, y_pred, labels)
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
    derived_pageids = y_true.keys()
    if not include_missing_predictions_in_derived_multilabel_metrics:
        derived_pageids = y_pred.keys()
    for pageid in derived_pageids:
        true_values = y_true[pageid]
        true_set = set(true_values)
        pred_set = set(y_pred.get(pageid, []))
        union = true_set | pred_set
        if union:
            jaccards.append(_safe_div(len(true_set & pred_set), len(union)))
        else:
            jaccards.append(1.0)
        hamming_errors += len(true_set ^ pred_set)
    metrics["jaccard"] = _safe_div(sum(jaccards), len(jaccards))

    if denominator_by_predicted:
        metrics["hamming_loss"] = _safe_div(hamming_errors, len(y_pred) * len(labels))
    else:
        total_labels = len(labels) if labels else 1
        metrics["hamming_loss"] = _safe_div(hamming_errors, max(len(y_true) * total_labels, 1))

    target_keys = [json.dumps(sorted(values), ensure_ascii=False) for values in y_true.values()]
    majority_key = Counter(target_keys).most_common(1)[0][0] if target_keys else "[]"
    majority_set = json.loads(majority_key)
    majority_pred = {pageid: list(majority_set) for pageid in y_true}
    empty_pred: dict[str, list[str]] = {pageid: [] for pageid in y_true}
    metrics["majority_labelset_exact_match_accuracy"] = multilabel_metrics(
        y_true, majority_pred, labels
    )["exact_match_accuracy"]
    metrics["empty_set_exact_match_accuracy"] = multilabel_metrics(y_true, empty_pred, labels)[
        "exact_match_accuracy"
    ]
    return metrics


def compute_task_subset_metrics(
    records: pd.DataFrame,
    *,
    task: str,
    labels: list[str],
) -> tuple[SpatialSubsetMetricResult, list[dict[str, Any]]]:
    if task == "corine_level2":
        return compute_single_label_subset_metrics(
            records,
            labels,
            include_records_without_target=False,
            include_missing_predictions=False,
        )
    metrics = compute_multilabel_subset_metrics(
        records,
        labels,
        require_list_targets=True,
        denominator_by_predicted=False,
    )
    return metrics, []
