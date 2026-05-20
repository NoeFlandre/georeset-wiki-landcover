import numpy as np

from georeset_wiki_landcover.analysis.supported_metrics import (
    per_label_counts,
    single_label_metrics_supported,
)


def test_supported_metrics_exclude_zero_support_labels_from_macro_averages() -> None:
    y_true = np.array(["21", "21", "22"])
    y_pred = np.array(["21", "33", "33"])
    labels = ["21", "22", "33"]

    metrics = single_label_metrics_supported(y_true, y_pred, labels)

    assert metrics["balanced_accuracy_supported"] == 0.25
    assert metrics["balanced_accuracy_allowed"] == (0.5 + 0.0 + 0.0) / 3.0
    assert metrics["macro_precision_supported"] == 0.5
    assert metrics["macro_f1_supported"] == 1.0 / 3.0
    assert metrics["macro_f1_allowed"] == 2.0 / 9.0


def test_per_label_counts_reports_predictions_for_zero_support_labels() -> None:
    counts = per_label_counts(["21", "22"], ["33", "22"], ["21", "22", "33"])

    assert counts["21"]["support"] == 1
    assert counts["22"]["true_positive"] == 1
    assert counts["33"]["support"] == 0
    assert counts["33"]["false_positive"] == 1
