import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.affinity import translate
from shapely.geometry import box

from scripts.reproduce_small import run_small_reproduction
from scripts.validate_artifacts import validate_artifacts


def test_small_reproduction_writes_valid_manifested_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"

    manifest = run_small_reproduction(output_dir=output_dir, clean=True)

    assert manifest["mode"] == "small"
    assert manifest["expected_counts"]["wiki_articles"] == 2
    assert manifest["expected_counts"]["corine_level2_summary_predictions"] == 2
    assert manifest["expected_counts"]["osm_summary_predictions"] == 2
    assert validate_artifacts(output_dir, profile="small") == []


def test_small_artifact_validator_reports_stale_manifest_hash(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    summaries_path = output_dir / "data/wiki/article_summaries.json"
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    summaries["100"]["summary"] = "Changed after manifest creation."
    summaries_path.write_text(json.dumps(summaries), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "stale artifact hash for data/wiki/article_summaries.json" in violation
        for violation in violations
    )


def test_small_artifact_validator_rejects_a_missing_required_manifest_hash(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_artifact = "data/wiki/article_contents.json"
    manifest["artifact_sha256"].pop(missing_artifact)
    assert manifest["artifact_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "artifact_sha256 missing required artifact hashes" in violation
        and missing_artifact in violation
        for violation in violations
    )


def test_small_artifact_validator_rejects_hashes_outside_the_artifact_inventory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    extra_path = output_dir / "extra.txt"
    extra_path.write_text("not a declared synthetic artifact", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["extra.txt"] = hashlib.sha256(extra_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "artifact_sha256" in violation and "extra.txt" in violation for violation in violations
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "unrecorded-model"),
        ("seed", 43),
        ("temperature", 0.5),
        ("classification_policy_version", "obsolete-policy"),
    ],
)
def test_small_validator_rejects_manifest_run_provenance_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any("manifest.json" in violation and field in violation for violation in violations)


@pytest.mark.parametrize(
    "field",
    ["workflow", "inputs", "outputs", "known_non_reproducible_components"],
)
def test_small_validator_rejects_manifest_artifact_provenance_drift(
    tmp_path: Path, field: str
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_values: dict[str, object] = {
        "workflow": "other-workflow",
        "inputs": manifest["inputs"][1:],
        "outputs": manifest["outputs"][:-1],
        "known_non_reproducible_components": ["unrecorded component"],
    }
    manifest[field] = changed_values[field]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any("manifest.json" in violation and field in violation for violation in violations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pageid", 100),
        ("title", "Unrelated title"),
        ("raw_response", '{"label": "21"}'),
        ("error", "unexpected error for a successful result"),
        ("unexpected", None),
    ],
)
def test_small_validator_rejects_drift_in_complete_prediction_records(
    tmp_path: Path, field: str, value: object
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    predictions_relative = "data/classification/runs/small/corine_level2_summary_predictions.json"
    predictions_path = output_dir / predictions_relative
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions["100"][field] = value
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    _refresh_manifest_hash(output_dir, predictions_relative)

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "synthetic" in violation
        and "corine_level2_summary" in violation
        and "100" in violation
        and field in violation
        for violation in violations
    )


def test_small_validator_rejects_drift_in_synthetic_source_titles(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    articles_relative = "data/wiki/wiki_articles.json"
    articles_path = output_dir / articles_relative
    articles = json.loads(articles_path.read_text(encoding="utf-8"))
    articles[0]["title"] = "Unrelated source title"
    articles_path.write_text(json.dumps(articles), encoding="utf-8")
    _refresh_manifest_hash(output_dir, articles_relative)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic wiki article 100 title" in violation for violation in violations)


def test_small_artifact_validator_reports_duplicate_wiki_pageids(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    wiki_path = output_dir / "data/wiki/wiki_articles.json"
    articles = json.loads(wiki_path.read_text(encoding="utf-8"))
    articles.append(dict(articles[0]))
    wiki_path.write_text(json.dumps(articles), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert "duplicate wiki pageids: 100" in violations


def _refresh_manifest_hash(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][relative_path] = hashlib.sha256(
        (root / relative_path).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_small_validator_rejects_a_self_consistent_missing_synthetic_prediction(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    predictions_relative = "data/classification/runs/small/corine_level2_summary_predictions.json"
    predictions_path = output_dir / predictions_relative
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions.pop("200")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    metrics_relative = "data/classification/runs/small/corine_level2_summary_metrics.json"
    metrics_path = output_dir / metrics_relative
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(n_eligible=1, n_predicted_ok=1, n_parse_error=0)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _refresh_manifest_hash(output_dir, predictions_relative)
    _refresh_manifest_hash(output_dir, metrics_relative)

    assert metrics["n_eligible"] == len(predictions)
    assert metrics["n_predicted_ok"] == len(predictions)
    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "200" in violation for violation in violations)


def test_small_validator_checks_the_known_synthetic_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    metrics_relative = "data/classification/runs/small/corine_level2_summary_metrics.json"
    metrics_path = output_dir / metrics_relative
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["accuracy"] = 0.5
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _refresh_manifest_hash(output_dir, metrics_relative)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "accuracy" in violation for violation in violations)


def test_small_validator_rejects_an_unexpected_null_metric_field(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    metrics_relative = "data/classification/runs/small/corine_level2_summary_metrics.json"
    metrics_path = output_dir / metrics_relative
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["unexpected"] = None
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _refresh_manifest_hash(output_dir, metrics_relative)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "metrics" in violation for violation in violations)


def test_small_validator_rejects_changed_synthetic_summary_with_refreshed_hash(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    relative_path = "data/wiki/article_summaries.json"
    summaries_path = output_dir / relative_path
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    summaries["100"]["summary"] = "Synthetic meadow land cover with grass."
    summaries_path.write_text(json.dumps(summaries), encoding="utf-8")
    _refresh_manifest_hash(output_dir, relative_path)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "summary" in violation for violation in violations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "different-model"),
        ("model_repo_id", "different/repository"),
        ("seed", 43),
        ("temperature", 0.5),
        ("allowed_labels", []),
        ("prompt", "different prompt"),
        ("system_prompt", "different system prompt"),
        ("attempt_count", 2),
        ("fingerprint", "0" * 64),
        ("text_sha256", "0" * 64),
    ],
)
def test_small_validator_rejects_changed_synthetic_prediction_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    relative_path = "data/classification/runs/small/corine_level2_summary_predictions.json"
    predictions_path = output_dir / relative_path
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions["100"]["metadata"][field] = value
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    _refresh_manifest_hash(output_dir, relative_path)

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "synthetic" in violation and "prediction 100" in violation and "metadata" in violation
        for violation in violations
    )


def test_small_validator_rejects_a_synthetic_wiki_point_moved_to_another_polygon(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    relative_path = "data/wiki/wiki_articles.json"
    wiki_path = output_dir / relative_path
    articles = json.loads(wiki_path.read_text(encoding="utf-8"))
    articles[0]["lon"] = 2.5
    wiki_path.write_text(json.dumps(articles), encoding="utf-8")
    _refresh_manifest_hash(output_dir, relative_path)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "coordinate" in violation for violation in violations)


def test_small_validator_rejects_changed_synthetic_corine_labels(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    relative_path = "data/corine/synthetic_corine.geojson"
    vector_path = output_dir / relative_path
    frame = gpd.read_file(vector_path)
    frame.loc[frame["code_18"] == "311", "code_18"] = "211"
    frame.to_file(vector_path, driver="GeoJSON", index=False)
    _refresh_manifest_hash(output_dir, relative_path)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "corine" in violation for violation in violations)


def test_small_validator_rejects_changed_synthetic_polygon_geometry(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    relative_path = "data/osm/osm_project_polygons.geojson"
    vector_path = output_dir / relative_path
    frame = gpd.read_file(vector_path)
    row = frame.index[frame["osm_id"] == "synthetic/wood"][0]
    frame.at[row, "geometry"] = translate(frame.at[row, "geometry"], xoff=0.25)
    frame.to_file(vector_path, driver="GeoJSON", index=False)
    _refresh_manifest_hash(output_dir, relative_path)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "geometry" in violation for violation in violations)


@pytest.mark.parametrize(
    ("field", "value"),
    [("target", "21"), ("prediction", "21"), ("prediction_labels", ["21"])],
)
def test_small_validator_rejects_prediction_payloads_that_disagree_with_known_metrics(
    tmp_path: Path, field: str, value: object
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    predictions_relative = "data/classification/runs/small/corine_level2_summary_predictions.json"
    predictions_path = output_dir / predictions_relative
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions["100"][field] = value
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    _refresh_manifest_hash(output_dir, predictions_relative)

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "synthetic" in violation and "prediction" in violation and "100" in violation
        for violation in violations
    )


@pytest.mark.parametrize(
    ("filename", "field_path"),
    [
        ("corine_level2_summary_metrics", "coverage"),
        ("corine_level2_summary_metrics", "macro_f1"),
        ("corine_level2_summary_metrics", "per_label.21.f1"),
        ("osm_summary_metrics", "coverage"),
        ("osm_summary_metrics", "macro_f1"),
        ("osm_summary_metrics", "per_label.meadow.f1"),
    ],
)
def test_small_validator_rejects_self_consistent_incomplete_metric_checks(
    tmp_path: Path, filename: str, field_path: str
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    metrics_relative = f"data/classification/runs/small/{filename}.json"
    metrics_path = output_dir / metrics_relative
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    target = metrics
    *parents, leaf = field_path.split(".")
    for part in parents:
        target = target[part]
    target[leaf] = 0.5
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _refresh_manifest_hash(output_dir, metrics_relative)

    violations = validate_artifacts(output_dir, profile="small")

    assert any("synthetic" in violation and "metrics" in violation for violation in violations)


@pytest.mark.parametrize("data_source", [None, "syntheti"])
def test_small_validator_requires_the_synthetic_manifest_marker(
    tmp_path: Path, data_source: str | None
) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_source is None:
        manifest.pop("data_source")
    else:
        manifest["data_source"] = data_source
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any("data_source" in violation and "synthetic" in violation for violation in violations)


def _write_full_artifacts(root: Path, pageid: int = 100) -> None:
    (root / "corine").mkdir(parents=True)
    (root / "wiki").mkdir(parents=True)
    (root / "osm").mkdir(parents=True)
    (root / "corine/bounds.json").write_text(
        json.dumps({"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0}),
        encoding="utf-8",
    )
    (root / "wiki/wiki_articles.json").write_text(
        json.dumps([{"pageid": pageid, "lat": 0.5, "lon": 0.5, "title": "Inside"}]),
        encoding="utf-8",
    )
    (root / "wiki/article_contents.json").write_text(
        json.dumps({str(pageid): {"content": "Inside"}}),
        encoding="utf-8",
    )
    osm = gpd.GeoDataFrame(
        {"osm_id": ["way/1"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    osm.to_file(root / "osm/osm_project_polygons.geojson", driver="GeoJSON")


def test_full_profile_accepts_non_synthetic_pageids(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_full_artifacts(root, pageid=300)

    assert validate_artifacts(root, profile="full") == []


def test_full_artifact_validator_reports_bad_bounds_schema(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_full_artifacts(root)
    (root / "corine/bounds.json").write_text(
        json.dumps({"min_lon": 0.0, "min_lat": 0.0, "max_lon": "east"}),
        encoding="utf-8",
    )

    violations = validate_artifacts(root, profile="full")

    assert (
        "corine/bounds.json must contain numeric min_lon, min_lat, max_lon, max_lat" in violations
    )


def test_full_artifact_validator_reports_malformed_wiki_rows(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_full_artifacts(root)
    (root / "wiki/wiki_articles.json").write_text(
        json.dumps(
            [
                {"lat": 0.5, "lon": 0.5, "title": "Missing pageid"},
                {"pageid": 100, "lat": "north", "lon": 0.5, "title": "Bad latitude"},
            ]
        ),
        encoding="utf-8",
    )

    violations = validate_artifacts(root, profile="full")

    assert "wiki article at index 0 missing pageid" in violations
    assert "wiki article 100 has non-numeric lat/lon" in violations
