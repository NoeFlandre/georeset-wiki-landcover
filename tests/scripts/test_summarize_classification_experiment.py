import csv
import json

from scripts.analysis.summarize_classification_experiment import (
    collect_metric_rows,
    majority_baseline_score,
    write_overview_csv,
    write_overview_markdown,
    write_readme,
)


def _write_metrics(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_predictions(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_majority_baseline_score_handles_single_and_multilabel_targets():
    corine_records = {
        "1": {"target": "31", "parse_status": "ok"},
        "2": {"target": "31", "parse_status": "ok"},
        "3": {"target": "21", "parse_status": "ok"},
    }
    osm_records = {
        "1": {"target": ["meadow"], "parse_status": "ok"},
        "2": {"target": ["meadow"], "parse_status": "ok"},
        "3": {"target": ["meadow", "wood"], "parse_status": "ok"},
    }

    assert majority_baseline_score(corine_records, "accuracy") == 2 / 3
    assert majority_baseline_score(osm_records, "exact_match_accuracy") == 2 / 3


def test_collect_metric_rows_sorts_and_selects_task_specific_scores(tmp_path):
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    _write_metrics(
        experiment_dir / "osm_summary_metrics.json",
        {
            "task": "osm",
            "text_source": "summary",
            "n_eligible": 2,
            "n_predicted_ok": 2,
            "n_parse_error": 0,
            "coverage": 1.0,
            "exact_match_accuracy": 0.5,
            "micro_precision": 0.75,
            "micro_recall": 0.85,
            "micro_f1": 0.8,
            "macro_precision": 0.65,
            "macro_recall": 0.75,
            "macro_f1": 0.7,
            "labels_evaluated": ["meadow", "wood"],
        },
    )
    _write_predictions(
        experiment_dir / "osm_summary_predictions.json",
        {
            "1": {"target": ["meadow"], "parse_status": "ok"},
            "2": {"target": ["meadow"], "parse_status": "ok"},
        },
    )
    _write_metrics(
        experiment_dir / "corine_level2_content_metrics.json",
        {
            "task": "corine_level2",
            "text_source": "content",
            "n_eligible": 3,
            "n_predicted_ok": 3,
            "n_parse_error": 0,
            "coverage": 1.0,
            "accuracy": 2 / 3,
            "macro_precision": 0.5,
            "macro_recall": 0.75,
            "macro_f1": 0.6,
            "labels_evaluated": ["21", "31"],
        },
    )
    _write_predictions(
        experiment_dir / "corine_level2_content_predictions.json",
        {
            "1": {"target": "31", "parse_status": "ok"},
            "2": {"target": "31", "parse_status": "ok"},
            "3": {"target": "21", "parse_status": "ok"},
        },
    )

    rows = collect_metric_rows(experiment_dir)

    assert [row["run"] for row in rows] == ["corine_level2/content", "osm/summary"]
    assert rows[0]["primary_score"] == 2 / 3
    assert rows[0]["majority_baseline_score"] == 2 / 3
    assert rows[0]["delta_vs_majority"] == 0
    assert rows[0]["accuracy"] == 2 / 3
    assert rows[0]["exact_match_accuracy"] == ""
    assert rows[0]["macro_precision"] == 0.5
    assert rows[0]["micro_f1"] == ""
    assert rows[1]["primary_score"] == 0.5
    assert rows[1]["majority_baseline_score"] == 1.0
    assert rows[1]["delta_vs_majority"] == -0.5
    assert rows[1]["exact_match_accuracy"] == 0.5
    assert rows[1]["micro_precision"] == 0.75
    assert rows[1]["macro_recall"] == 0.75
    assert rows[1]["micro_f1"] == 0.8
    assert rows[1]["n_labels_evaluated"] == 2


def test_write_overview_outputs_csv_and_markdown(tmp_path):
    rows = [
        {
            "run": "osm/summary",
            "task": "osm",
            "text_source": "summary",
            "n_eligible": 2,
            "n_predicted_ok": 2,
            "n_parse_error": 0,
            "coverage": 1.0,
            "primary_metric": "exact_match_accuracy",
            "primary_score": 0.5,
            "majority_baseline_score": 0.4,
            "delta_vs_majority": 0.1,
            "accuracy": "",
            "exact_match_accuracy": 0.5,
            "macro_precision": 0.6,
            "macro_recall": 0.7,
            "macro_f1": 0.7,
            "micro_precision": 0.75,
            "micro_recall": 0.85,
            "micro_f1": 0.8,
            "n_labels_evaluated": 2,
        }
    ]
    csv_path = tmp_path / "overview.csv"
    md_path = tmp_path / "overview.md"

    write_overview_csv(rows, csv_path)
    write_overview_markdown(rows, md_path)

    with csv_path.open() as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["run"] == "osm/summary"
    assert csv_rows[0]["primary_metric"] == "exact_match_accuracy"
    assert csv_rows[0]["majority_baseline_score"] == "0.4"
    assert csv_rows[0]["delta_vs_majority"] == "0.1"
    assert csv_rows[0]["micro_precision"] == "0.75"
    assert "majority baseline" in md_path.read_text(encoding="utf-8")
    assert "micro precision" in md_path.read_text(encoding="utf-8")
    assert "osm/summary" in md_path.read_text(encoding="utf-8")


def test_write_readme_describes_experiment_inputs_and_outputs(tmp_path):
    rows = [
        {
            "run": "corine_level2/summary",
            "task": "corine_level2",
            "text_source": "summary",
            "n_eligible": 1251,
            "n_predicted_ok": 1251,
            "n_parse_error": 0,
            "coverage": 1.0,
            "primary_metric": "accuracy",
            "primary_score": 0.23,
            "majority_baseline_score": 0.2,
            "delta_vs_majority": 0.03,
            "accuracy": 0.23,
            "exact_match_accuracy": "",
            "macro_precision": 0.25,
            "macro_recall": 0.26,
            "macro_f1": 0.24,
            "micro_precision": "",
            "micro_recall": "",
            "micro_f1": "",
            "n_labels_evaluated": 9,
        }
    ]
    readme_path = tmp_path / "README.md"

    write_readme(rows, readme_path)

    text = readme_path.read_text(encoding="utf-8")
    assert "Article-Text Classification E2E v1" in text
    assert "2 tasks x 3 text sources" in text
    assert "`summary`: generated with `summary_mode=place`" in text
    assert "overview.csv" in text
    assert "corine_level2/summary" in text
    assert "Metric Explanations" in text
    assert "best delta vs majority baseline" in text
    assert "meadow" in text
    assert "water" in text
    assert "100x more" in text
    assert "`n_eligible`" in text
    assert "`exact_match_accuracy`" in text
    assert "`majority_baseline_score`" in text
