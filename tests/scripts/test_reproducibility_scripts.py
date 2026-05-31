import json
from pathlib import Path

import geopandas as gpd
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


def test_small_artifact_validator_reports_duplicate_wiki_pageids(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    wiki_path = output_dir / "data/wiki/wiki_articles.json"
    articles = json.loads(wiki_path.read_text(encoding="utf-8"))
    articles.append(dict(articles[0]))
    wiki_path.write_text(json.dumps(articles), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert "duplicate wiki pageids: 100" in violations


def _write_full_artifacts(root: Path) -> None:
    (root / "corine").mkdir(parents=True)
    (root / "wiki").mkdir(parents=True)
    (root / "osm").mkdir(parents=True)
    (root / "corine/bounds.json").write_text(
        json.dumps({"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0}),
        encoding="utf-8",
    )
    (root / "wiki/wiki_articles.json").write_text(
        json.dumps([{"pageid": 100, "lat": 0.5, "lon": 0.5, "title": "Inside"}]),
        encoding="utf-8",
    )
    (root / "wiki/article_contents.json").write_text(
        json.dumps({"100": {"content": "Inside"}}),
        encoding="utf-8",
    )
    osm = gpd.GeoDataFrame(
        {"osm_id": ["way/1"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    osm.to_file(root / "osm/osm_project_polygons.geojson", driver="GeoJSON")


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
