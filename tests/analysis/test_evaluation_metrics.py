"""Tests for reusable classification subset metric helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from georeset.analysis.evaluation_metrics import (
    compute_multilabel_subset_metrics,
    compute_single_label_subset_metrics,
    compute_task_subset_metrics,
)


def _as_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pageid": "1", "target": "31", "prediction": "31", "parse_status": "ok"},
            {"pageid": "2", "target": "21", "prediction": "31", "parse_status": "ok"},
            {"pageid": "3", "target": "31", "prediction": "31", "parse_status": "error"},
            {"pageid": "4", "target": "31", "prediction": "21", "parse_status": "error"},
        ]
    )


def test_single_label_subset_metrics_compute_expected_fields() -> None:
    records = _as_records()
    metrics, per_class = compute_single_label_subset_metrics(
        records,
        labels=["21", "31"],
        include_records_without_target=True,
        include_missing_predictions=False,
    )

    assert metrics["n"] == 4
    assert metrics["n_predicted_ok"] == 2
    assert metrics["n_parse_error"] == 2
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_precision"] == pytest.approx(0.25)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.3333333333333333)
    assert metrics["weighted_precision"] == pytest.approx(0.375)
    assert metrics["weighted_recall"] == pytest.approx(0.75)
    assert metrics["weighted_f1"] == pytest.approx(0.5)
    assert metrics["majority_accuracy"] == pytest.approx(0.75)
    assert metrics["majority_balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["majority_macro_f1"] == pytest.approx(0.42857142857142855)
    assert metrics["delta_vs_majority_accuracy"] == pytest.approx(-0.25)
    assert metrics["delta_vs_majority_balanced_accuracy"] == pytest.approx(0.0)
    assert metrics["delta_vs_majority_macro_f1"] == pytest.approx(-0.09523809523809523)

    assert len(per_class) == 2
    by_label = {row["label"]: row for row in per_class}
    assert by_label["21"]["support"] == 1
    assert by_label["21"]["precision"] == pytest.approx(0.0)
    assert by_label["21"]["recall"] == pytest.approx(0.0)
    assert by_label["21"]["f1"] == pytest.approx(0.0)
    assert by_label["31"]["support"] == 3
    assert by_label["31"]["precision"] == pytest.approx(0.5)
    assert by_label["31"]["recall"] == pytest.approx(1.0)
    assert by_label["31"]["f1"] == pytest.approx(0.6666666666666666)


def test_single_label_subset_metrics_treats_missing_or_parse_error_predictions_as_parse_errors() -> (
    None
):
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": "a", "prediction": "a", "parse_status": "ok"},
            {"pageid": "2", "target": "b", "prediction": None, "parse_status": "ok"},
            {"pageid": "3", "target": "b", "prediction": None, "parse_status": "error"},
            {"pageid": "4", "target": "a", "prediction": "b", "parse_status": "error"},
        ]
    )
    metrics, _ = compute_single_label_subset_metrics(
        records,
        labels=["a", "b"],
        include_records_without_target=True,
        include_missing_predictions=False,
    )

    assert metrics["n"] == 4
    assert metrics["n_predicted_ok"] == 1
    assert metrics["n_parse_error"] == 3
    assert metrics["coverage"] == pytest.approx(0.25)
    # pageid 2 has missing prediction; only row 1 is evaluated.
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_multilabel_subset_metrics_compute_expected_fields() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
            {"pageid": "2", "target": ["water"], "prediction": ["wood"], "parse_status": "ok"},
            {
                "pageid": "3",
                "target": ["wood", "water"],
                "prediction": None,
                "parse_status": "error",
            },
            {"pageid": "4", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        ]
    )

    metrics = compute_multilabel_subset_metrics(
        records,
        labels=["wood", "water", "urban"],
        require_list_targets=True,
        denominator_by_predicted=True,
        include_missing_predictions_in_derived_multilabel_metrics=True,
    )

    assert metrics["n"] == 4
    assert metrics["n_predicted_ok"] == 3
    assert metrics["n_parse_error"] == 1
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["exact_match_accuracy"] == pytest.approx(2 / 3)
    assert metrics["micro_precision"] == pytest.approx(2 / 3)
    assert metrics["micro_recall"] == pytest.approx(2 / 3)
    assert metrics["micro_f1"] == pytest.approx(0.6666666666666666)
    assert metrics["macro_precision"] == pytest.approx(0.2222222222222222)
    assert metrics["macro_recall"] == pytest.approx(0.3333333333333333)
    assert metrics["macro_f1"] == pytest.approx(0.26666666666666666)
    assert metrics["jaccard"] == pytest.approx(0.5)
    assert metrics["hamming_loss"] == pytest.approx(4 / 9)
    assert metrics["majority_labelset_exact_match_accuracy"] == pytest.approx(0.5)
    assert metrics["empty_set_exact_match_accuracy"] == pytest.approx(0.0)


def test_multilabel_subset_metrics_counts_missing_as_empty_for_derived_metrics() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
            {"pageid": "2", "target": ["water"], "prediction": ["wood"], "parse_status": "ok"},
            {
                "pageid": "3",
                "target": ["wood", "water"],
                "prediction": None,
                "parse_status": "error",
            },
            {"pageid": "4", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        ]
    )
    metrics = compute_multilabel_subset_metrics(
        records,
        labels=["wood", "water", "urban"],
        require_list_targets=True,
        denominator_by_predicted=True,
        include_missing_predictions_in_derived_multilabel_metrics=True,
    )

    assert metrics["jaccard"] == pytest.approx(0.5)
    assert metrics["hamming_loss"] == pytest.approx(4 / 9)


def test_multilabel_subset_metrics_skips_missing_for_derived_metrics() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
            {"pageid": "2", "target": ["water"], "prediction": ["wood"], "parse_status": "ok"},
            {
                "pageid": "3",
                "target": ["wood", "water"],
                "prediction": None,
                "parse_status": "error",
            },
            {"pageid": "4", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        ]
    )
    metrics = compute_multilabel_subset_metrics(
        records,
        labels=["wood", "water", "urban"],
        require_list_targets=True,
        denominator_by_predicted=True,
        include_missing_predictions_in_derived_multilabel_metrics=False,
    )

    assert metrics["jaccard"] == pytest.approx(2 / 3)
    assert metrics["hamming_loss"] == pytest.approx(2 / 9)


def test_multilabel_subset_metrics_supported_denominator_switch() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
            {"pageid": "2", "target": ["water"], "prediction": ["wood"], "parse_status": "ok"},
            {
                "pageid": "3",
                "target": ["wood", "water"],
                "prediction": None,
                "parse_status": "error",
            },
            {"pageid": "4", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        ]
    )
    metrics = compute_multilabel_subset_metrics(
        records,
        labels=[],
        require_list_targets=True,
        denominator_by_predicted=False,
    )

    assert metrics["hamming_loss"] == 1.0


def test_compute_task_subset_metrics_dispatches_corine_and_returns_per_class() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": "31", "prediction": "31", "parse_status": "ok"},
            {"pageid": "2", "target": "21", "prediction": "31", "parse_status": "ok"},
        ]
    )

    metrics, per_class = compute_task_subset_metrics(
        records,
        task="corine_level2",
        labels=["21", "31"],
    )

    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert [row["label"] for row in per_class] == ["21", "31"]


def test_compute_task_subset_metrics_dispatches_multilabel_without_per_class() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
            {"pageid": "2", "target": ["water"], "prediction": ["wood"], "parse_status": "ok"},
        ]
    )

    metrics, per_class = compute_task_subset_metrics(
        records,
        task="osm",
        labels=["water", "wood"],
    )

    assert metrics["jaccard"] == pytest.approx(0.5)
    assert per_class == []


def test_compute_task_subset_metrics_rejects_unknown_task() -> None:
    records = pd.DataFrame(
        [
            {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        ]
    )

    with pytest.raises(ValueError, match="Unsupported classification task"):
        compute_task_subset_metrics(records, task="unknown", labels=["wood"])
