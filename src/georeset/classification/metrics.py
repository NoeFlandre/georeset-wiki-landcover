from georeset.contracts import MultiLabelMetricResult, PerLabelMetric, SingleLabelMetricResult
from georeset.utils.math import safe_div


def single_label_metrics(
    y_true: dict[str, str], y_pred: dict[str, str], labels: list[str]
) -> SingleLabelMetricResult:
    n_eligible = len(y_true)
    evaluated = {k: v for k, v in y_pred.items() if k in y_true}
    evaluated_n = len(evaluated)
    n_parse_error = n_eligible - evaluated_n
    coverage = safe_div(evaluated_n, n_eligible)
    accuracy = safe_div(
        sum(1 for k, v in evaluated.items() if y_true[k] == v),
        evaluated_n,
    )
    evaluated_true = {k: y_true[k] for k in evaluated}
    per_label: dict[str, PerLabelMetric] = {}
    for label in labels:
        tp = sum(1 for k, v in evaluated.items() if y_true[k] == label and v == label)
        fp = sum(1 for k, v in evaluated.items() if y_true[k] != label and v == label)
        fn = sum(
            1
            for k, true_label in evaluated_true.items()
            if true_label == label and evaluated[k] != label
        )
        support = sum(1 for k, v in y_true.items() if v == label)
        per_label[label] = {
            "support": support,
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        }
    macro_precision = safe_div(sum(p["precision"] for p in per_label.values()), len(labels))
    macro_recall = safe_div(sum(p["recall"] for p in per_label.values()), len(labels))
    macro_f1 = safe_div(sum(p["f1"] for p in per_label.values()), len(labels))
    return {
        "n_eligible": n_eligible,
        "n_predicted_ok": evaluated_n,
        "n_parse_error": n_parse_error,
        "coverage": coverage,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def multilabel_metrics(
    y_true: dict[str, list[str]], y_pred: dict[str, list[str]], labels: list[str]
) -> MultiLabelMetricResult:
    n_eligible = len(y_true)
    evaluated = {k: v for k, v in y_pred.items() if k in y_true}
    evaluated_n = len(evaluated)
    n_parse_error = n_eligible - evaluated_n
    coverage = safe_div(evaluated_n, n_eligible)
    exact_match = safe_div(
        sum(1 for k, v in evaluated.items() if set(y_true[k]) == set(v)),
        evaluated_n,
    )
    micro_tp = sum(1 for k, v in evaluated.items() for label in set(y_true[k]) & set(v))
    micro_fp = sum(1 for k, v in evaluated.items() for label in set(v) - set(y_true[k]))
    micro_fn = sum(1 for k, v in evaluated.items() for label in set(y_true[k]) - set(v))
    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn)
    per_label: dict[str, PerLabelMetric] = {}
    for label in labels:
        tp = sum(1 for k, v in evaluated.items() if label in y_true[k] and label in v)
        fp = sum(1 for k, v in evaluated.items() if label not in y_true[k] and label in v)
        fn = sum(1 for k, v in evaluated.items() if label in y_true[k] and label not in v)
        support = sum(1 for k, v in y_true.items() if label in v)
        per_label[label] = {
            "support": support,
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        }
    macro_precision = safe_div(sum(p["precision"] for p in per_label.values()), len(labels))
    macro_recall = safe_div(sum(p["recall"] for p in per_label.values()), len(labels))
    macro_f1 = safe_div(sum(p["f1"] for p in per_label.values()), len(labels))
    return {
        "n_eligible": n_eligible,
        "n_predicted_ok": evaluated_n,
        "n_parse_error": n_parse_error,
        "coverage": coverage,
        "exact_match_accuracy": exact_match,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
    }
