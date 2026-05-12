import csv
import json
from pathlib import Path

import pandas as pd

from georeset.cli.analysis.evaluate_predictions_with_spatial_confidence import main


def _write_prediction(path: Path, records: dict[str, dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _make_parent(path: Path) -> None:
    path.mkdir()
    corine_base: dict[str, dict[str, object]] = {
        "1": {"pageid": "1", "target": "31", "prediction": "31", "parse_status": "ok"},
        "2": {"pageid": "2", "target": "21", "prediction": "31", "parse_status": "ok"},
        "3": {"pageid": "3", "target": "31", "prediction": "21", "parse_status": "ok"},
    }
    corine_shuffled: dict[str, dict[str, object]] = {
        "1": {"pageid": "1", "target": "31", "prediction": "21", "parse_status": "ok"},
        "2": {"pageid": "2", "target": "21", "prediction": "21", "parse_status": "ok"},
        "3": {"pageid": "3", "target": "31", "prediction": "21", "parse_status": "ok"},
    }
    osm_base: dict[str, dict[str, object]] = {
        "1": {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        "2": {"pageid": "2", "target": ["water"], "prediction": [], "parse_status": "ok"},
        "4": {"pageid": "4", "target": ["wood"], "prediction": ["water"], "parse_status": "ok"},
    }
    osm_shuffled: dict[str, dict[str, object]] = {
        "1": {"pageid": "1", "target": ["wood"], "prediction": [], "parse_status": "ok"},
        "2": {"pageid": "2", "target": ["water"], "prediction": ["water"], "parse_status": "ok"},
        "4": {"pageid": "4", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
    }
    for source in ["summary", "summary_no_place", "content"]:
        _write_prediction(path / f"corine_level2_{source}_predictions.json", corine_base)
        _write_prediction(path / f"osm_{source}_predictions.json", osm_base)
    for source in ["summary_shuffled", "summary_no_place_shuffled", "content_shuffled"]:
        _write_prediction(path / f"corine_level2_{source}_predictions.json", corine_shuffled)
        _write_prediction(path / f"osm_{source}_predictions.json", osm_shuffled)


def _make_parent_with_parse_errors(path: Path) -> None:
    path.mkdir()
    corine: dict[str, dict[str, object]] = {
        "1": {"pageid": "1", "target": "31", "prediction": "31", "parse_status": "ok"},
        "2": {"pageid": "2", "target": "21", "prediction": None, "parse_status": "error"},
        "3": {"pageid": "3", "target": "31", "prediction": "21", "parse_status": "ok"},
    }
    osm: dict[str, dict[str, object]] = {
        "1": {"pageid": "1", "target": ["wood"], "prediction": ["wood"], "parse_status": "ok"},
        "2": {"pageid": "2", "target": ["water"], "prediction": None, "parse_status": "error"},
        "4": {"pageid": "4", "target": ["wood"], "prediction": ["water"], "parse_status": "ok"},
    }
    for source in [
        "summary",
        "summary_no_place",
        "content",
        "summary_shuffled",
        "summary_no_place_shuffled",
        "content_shuffled",
    ]:
        _write_prediction(path / f"corine_level2_{source}_predictions.json", corine)
        _write_prediction(path / f"osm_{source}_predictions.json", osm)


def _write_spatial(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "pageid": "1",
                "point_label": "31",
                "point_label_share_250m": 0.95,
                "point_label_share_500m": 0.85,
                "dominant_matches_point_label_250m": True,
                "dominant_matches_point_label_500m": True,
            },
            {
                "pageid": "2",
                "point_label": "21",
                "point_label_share_250m": 0.75,
                "point_label_share_500m": 0.82,
                "dominant_matches_point_label_250m": False,
                "dominant_matches_point_label_500m": True,
            },
            {
                "pageid": "3",
                "point_label": "31",
                "point_label_share_250m": 0.40,
                "point_label_share_500m": 0.30,
                "dominant_matches_point_label_250m": False,
                "dominant_matches_point_label_500m": False,
            },
        ]
    ).to_csv(path, index=False)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_spatial_subset_evaluation_recomputes_metrics_and_preserves_parent(tmp_path):
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    spatial_path = tmp_path / "spatial.csv"
    marker_path = parent_dir / "README.md"

    _make_parent(parent_dir)
    marker_path.write_text("frozen", encoding="utf-8")
    _write_spatial(spatial_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert marker_path.read_text(encoding="utf-8") == "frozen"
    for filename in [
        "overview_spatial_subsets.csv",
        "subset_counts.csv",
        "shuffled_delta_spatial_subsets.csv",
        "majority_baselines_spatial_subsets.csv",
        "per_class_metrics_corine_spatial_subsets.csv",
        "class_distribution_by_spatial_subset.csv",
        "manifest.json",
        "summary.md",
    ]:
        assert (output_dir / filename).exists()

    overview = _read_csv(output_dir / "overview_spatial_subsets.csv")
    corine_high = next(
        row
        for row in overview
        if row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["subset"] == "point_label_share_250m_ge_0.8"
    )
    assert corine_high["n"] == "1"
    assert corine_high["accuracy"] == "1.0"
    assert corine_high["majority_accuracy"] == "1.0"

    all_osm = next(
        row
        for row in overview
        if row["task"] == "osm"
        and row["text_source"] == "summary"
        and row["subset"] == "all_available_spatial_confidence"
    )
    assert all_osm["n"] == "2"
    assert float(all_osm["exact_match_accuracy"]) == 0.5

    deltas = _read_csv(output_dir / "shuffled_delta_spatial_subsets.csv")
    assert any(
        row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["shuffled_text_source"] == "summary_shuffled"
        and row["subset"] == "all_available_spatial_confidence"
        for row in deltas
    )

    distribution = _read_csv(output_dir / "class_distribution_by_spatial_subset.csv")
    assert any(
        row["task"] == "corine_level2"
        and row["subset"] == "all_available_spatial_confidence"
        and row["label"] == "31"
        and row["support"] == "2"
        for row in distribution
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["parent_experiment_id"]
        == "article_text_classification_e2e_with_shuffled_control_v1"
    )
    assert manifest["spatial_confidence_experiment_id"] == "corine_spatial_confidence_v1"
    assert manifest["no_llm_rerun"] is True
    assert manifest["osm_spatial_join_coverage"]["n_osm_predictions_total"] == 18
    assert manifest["osm_spatial_join_coverage"]["n_osm_predictions_with_spatial_confidence"] == 12
    assert (
        manifest["osm_spatial_join_coverage"]["n_osm_predictions_missing_spatial_confidence"] == 6
    )


def test_spatial_subset_evaluation_accepts_experiment_id_overrides_without_changing_metrics(
    tmp_path,
):
    parent_dir = tmp_path / "parent"
    default_output_dir = tmp_path / "out_default"
    custom_output_dir = tmp_path / "out_custom"
    spatial_path = tmp_path / "spatial.csv"

    _make_parent(parent_dir)
    _write_spatial(spatial_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(default_output_dir),
        ]
    )
    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(custom_output_dir),
            "--experiment-id",
            "custom_spatial_eval",
            "--parent-experiment-id",
            "custom_parent",
            "--spatial-confidence-experiment-id",
            "custom_spatial_confidence",
        ]
    )

    default_manifest = json.loads(
        (default_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    custom_manifest = json.loads((custom_output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert default_manifest["experiment_id"] == "article_text_classification_spatial_confidence_v1"
    assert (
        default_manifest["parent_experiment_id"]
        == "article_text_classification_e2e_with_shuffled_control_v1"
    )
    assert default_manifest["spatial_confidence_experiment_id"] == "corine_spatial_confidence_v1"
    assert custom_manifest["experiment_id"] == "custom_spatial_eval"
    assert custom_manifest["parent_experiment_id"] == "custom_parent"
    assert custom_manifest["spatial_confidence_experiment_id"] == "custom_spatial_confidence"

    summary = (custom_output_dir / "summary.md").read_text(encoding="utf-8")
    assert "custom_spatial_eval" in summary
    assert "custom_parent" in summary
    assert "custom_spatial_confidence" in summary

    assert (
        (default_output_dir / "overview_spatial_subsets.csv").read_text(encoding="utf-8")
        == (custom_output_dir / "overview_spatial_subsets.csv").read_text(encoding="utf-8")
    )
    assert (
        (default_output_dir / "shuffled_delta_spatial_subsets.csv").read_text(encoding="utf-8")
        == (custom_output_dir / "shuffled_delta_spatial_subsets.csv").read_text(encoding="utf-8")
    )


def test_spatial_subset_evaluation_treats_non_ok_predictions_as_parse_errors(tmp_path):
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "out"
    spatial_path = tmp_path / "spatial.csv"

    _make_parent_with_parse_errors(parent_dir)
    _write_spatial(spatial_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--spatial-confidence-path",
            str(spatial_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    overview = _read_csv(output_dir / "overview_spatial_subsets.csv")
    corine_all = next(
        row
        for row in overview
        if row["task"] == "corine_level2"
        and row["text_source"] == "summary"
        and row["subset"] == "all_available_spatial_confidence"
    )
    assert corine_all["n"] == "3"
    assert corine_all["n_predicted_ok"] == "2"
    assert corine_all["n_parse_error"] == "1"
    assert corine_all["coverage"] == "0.6666666666666666"
    assert corine_all["accuracy"] == "0.5"

    osm_all = next(
        row
        for row in overview
        if row["task"] == "osm"
        and row["text_source"] == "summary"
        and row["subset"] == "all_available_spatial_confidence"
    )
    assert osm_all["n"] == "2"
    assert osm_all["n_predicted_ok"] == "1"
    assert osm_all["n_parse_error"] == "1"
    assert osm_all["coverage"] == "0.5"
    assert osm_all["exact_match_accuracy"] == "1.0"
