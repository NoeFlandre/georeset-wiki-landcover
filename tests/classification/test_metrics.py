from georeset.classification.metrics import multilabel_metrics, single_label_metrics


def test_single_label_metrics_computes_accuracy_and_macro_scores():
    y_true = {"1": "21", "2": "31", "3": "31"}
    y_pred = {"1": "21", "2": "21", "3": "31"}
    metrics = single_label_metrics(y_true, y_pred, labels=["21", "31"])
    assert metrics["n_eligible"] == 3
    assert metrics["n_predicted_ok"] == 3
    assert metrics["n_parse_error"] == 0
    assert metrics["coverage"] == 1.0
    assert metrics["accuracy"] == 2 / 3
    assert round(metrics["macro_f1"], 4) == 0.6667
    assert metrics["per_label"]["21"]["support"] == 1
    assert metrics["per_label"]["31"]["support"] == 2


def test_single_label_metrics_handles_parse_errors():
    y_true = {"1": "21", "2": "31"}
    y_pred = {"1": "21"}  # article "2" has no prediction (parse error)
    metrics = single_label_metrics(y_true, y_pred, labels=["21", "31"])
    assert metrics["n_eligible"] == 2
    assert metrics["n_predicted_ok"] == 1
    assert metrics["n_parse_error"] == 1
    assert metrics["coverage"] == 0.5
    # accuracy is over the one evaluated prediction: "1" predicted and true both "21"
    assert metrics["accuracy"] == 1.0
    # macro_f1: only label "21" has TP=1, FP=0, FN=0; "31" has TP=0,FP=0,FN=1
    assert round(metrics["per_label"]["21"]["f1"], 4) == 1.0
    assert metrics["per_label"]["31"]["recall"] == 0.0


def test_multilabel_metrics_computes_exact_match_and_micro_scores():
    y_true = {"1": ["meadow", "wood"], "2": ["water"]}
    y_pred = {"1": ["meadow"], "2": ["water"]}
    metrics = multilabel_metrics(y_true, y_pred, labels=["meadow", "water", "wood"])
    assert metrics["n_eligible"] == 2
    assert metrics["n_predicted_ok"] == 2
    assert metrics["n_parse_error"] == 0
    assert metrics["coverage"] == 1.0
    assert metrics["exact_match_accuracy"] == 0.5
    assert round(metrics["micro_precision"], 4) == 1.0
    assert round(metrics["micro_recall"], 4) == 0.6667
    assert round(metrics["micro_f1"], 4) == 0.8
