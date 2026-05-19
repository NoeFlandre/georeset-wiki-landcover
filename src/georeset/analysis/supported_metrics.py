"""Single-label metrics with explicit allowed-label and supported-label averages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class LabelCounts(TypedDict):
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int


def _as_string_array(values: Iterable[object]) -> NDArray[np.str_]:
    return np.asarray([str(value) for value in values])


def per_label_counts(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    labels: Sequence[str],
) -> dict[str, LabelCounts]:
    true = _as_string_array(y_true)
    pred = _as_string_array(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    result: dict[str, LabelCounts] = {}
    for label in labels:
        label_true = true == label
        label_pred = pred == label
        true_positive = int((label_true & label_pred).sum())
        false_positive = int((~label_true & label_pred).sum())
        false_negative = int((label_true & ~label_pred).sum())
        result[str(label)] = {
            "support": int(label_true.sum()),
            "predicted": int(label_pred.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    return result


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def single_label_metrics_supported(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    labels: Sequence[str],
) -> dict[str, float]:
    """Compute single-label metrics over allowed labels and labels with true support."""
    true = _as_string_array(y_true)
    pred = _as_string_array(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    counts = per_label_counts(true, pred, labels)
    precision_by_label: list[float] = []
    recall_by_label: list[float] = []
    f1_by_label: list[float] = []
    supported_precision: list[float] = []
    supported_recall: list[float] = []
    supported_f1: list[float] = []
    weighted_f1_numerator = 0.0
    total_support = 0
    for values in counts.values():
        precision = _safe_div(
            values["true_positive"],
            values["true_positive"] + values["false_positive"],
        )
        recall = _safe_div(
            values["true_positive"],
            values["true_positive"] + values["false_negative"],
        )
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        precision_by_label.append(precision)
        recall_by_label.append(recall)
        f1_by_label.append(f1)
        if values["support"] > 0:
            supported_precision.append(precision)
            supported_recall.append(recall)
            supported_f1.append(f1)
            weighted_f1_numerator += f1 * values["support"]
            total_support += values["support"]

    return {
        "accuracy": float((true == pred).mean()) if len(true) else 0.0,
        "balanced_accuracy_allowed": float(np.mean(recall_by_label)) if recall_by_label else 0.0,
        "balanced_accuracy_supported": float(np.mean(supported_recall))
        if supported_recall
        else 0.0,
        "macro_precision_allowed": float(np.mean(precision_by_label))
        if precision_by_label
        else 0.0,
        "macro_precision_supported": float(np.mean(supported_precision))
        if supported_precision
        else 0.0,
        "macro_f1_allowed": float(np.mean(f1_by_label)) if f1_by_label else 0.0,
        "macro_f1_supported": float(np.mean(supported_f1)) if supported_f1 else 0.0,
        "weighted_f1_supported": _safe_div(weighted_f1_numerator, total_support),
    }
