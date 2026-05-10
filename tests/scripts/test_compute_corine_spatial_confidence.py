import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from georeset.cli.data.compute_corine_spatial_confidence import main


def _write_prediction(path: Path, records: dict[str, dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _make_parent_experiment(path: Path, *, mismatch: bool = False) -> None:
    path.mkdir()
    corine_records = {
        "1": {"pageid": "1", "target": "31", "parse_status": "ok"},
        "2": {"pageid": "2", "target": "21" if mismatch else "31", "parse_status": "ok"},
    }
    osm_records = {
        "1": {"pageid": "1", "target": ["wood"], "parse_status": "ok"},
        "3": {"pageid": "3", "target": ["water"], "parse_status": "ok"},
    }
    for text_source in [
        "summary",
        "summary_no_place",
        "content",
        "summary_shuffled",
        "summary_no_place_shuffled",
        "content_shuffled",
    ]:
        _write_prediction(path / f"corine_level2_{text_source}_predictions.json", corine_records)
        _write_prediction(path / f"osm_{text_source}_predictions.json", osm_records)


def _write_articles(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"pageid": 1, "lat": 48.0, "lon": 7.0, "title": "A"},
                {"pageid": 2, "lat": 48.001, "lon": 7.001, "title": "B"},
                {"pageid": 3, "lat": 48.002, "lon": 7.002, "title": "C"},
            ]
        ),
        encoding="utf-8",
    )


def _write_corine(path: Path) -> None:
    articles = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([7.0, 7.001, 7.002], [48.0, 48.001, 48.002]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    geoms = [point.buffer(20).envelope for point in articles.geometry]
    gdf = gpd.GeoDataFrame(
        {"code_18": ["311", "311", "112"]},
        geometry=geoms,
        crs="EPSG:2154",
    ).to_crs("EPSG:4326")
    gdf.to_file(path)


def _write_corine_missing_parent_target(path: Path) -> None:
    articles = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([7.0, 7.002], [48.0, 48.002]),
        crs="EPSG:4326",
    ).to_crs("EPSG:2154")
    geoms = [point.buffer(20).envelope for point in articles.geometry]
    gdf = gpd.GeoDataFrame(
        {"code_18": ["311", "112"]},
        geometry=geoms,
        crs="EPSG:2154",
    ).to_crs("EPSG:4326")
    gdf.to_file(path)


def test_cli_computes_spatial_confidence_for_union_of_parent_prediction_pageids(tmp_path):
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "spatial"
    wiki_path = tmp_path / "wiki.json"
    corine_path = tmp_path / "corine.geojson"
    untouched = parent_dir / "README.md"

    _make_parent_experiment(parent_dir)
    untouched.write_text("do not change", encoding="utf-8")
    _write_articles(wiki_path)
    _write_corine(corine_path)

    main(
        [
            "--parent-experiment-dir",
            str(parent_dir),
            "--wiki-articles-path",
            str(wiki_path),
            "--corine-polygons-path",
            str(corine_path),
            "--output-dir",
            str(output_dir),
            "--radii",
            "250",
            "500",
        ]
    )

    confidence_path = output_dir / "spatial_confidence.csv"
    manifest_path = output_dir / "manifest.json"
    assert confidence_path.exists()
    assert manifest_path.exists()
    assert untouched.read_text(encoding="utf-8") == "do not change"

    confidence = pd.read_csv(confidence_path, dtype={"pageid": str, "point_label": str})
    assert set(confidence["pageid"]) == {"1", "2", "3"}
    assert {
        "point_label",
        "point_label_share_250m",
        "dominant_label_250m",
        "artificial_share_250m",
        "artificial_share_500m",
    }.issubset(confidence.columns)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "corine_spatial_confidence_v1"
    assert (
        manifest["parent_experiment_id"]
        == "article_text_classification_e2e_with_shuffled_control_v1"
    )
    assert manifest["used_full_corine"] is True
    assert manifest["number_of_articles"] == 3


def test_cli_fails_loudly_when_derived_point_label_disagrees_with_corine_target(tmp_path):
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "spatial"
    wiki_path = tmp_path / "wiki.json"
    corine_path = tmp_path / "corine.geojson"

    _make_parent_experiment(parent_dir, mismatch=True)
    _write_articles(wiki_path)
    _write_corine(corine_path)

    with pytest.raises(ValueError, match="CORINE target mismatch"):
        main(
            [
                "--parent-experiment-dir",
                str(parent_dir),
                "--wiki-articles-path",
                str(wiki_path),
                "--corine-polygons-path",
                str(corine_path),
                "--output-dir",
                str(output_dir),
                "--radii",
                "250",
            ]
        )


def test_cli_fails_loudly_when_parent_corine_target_has_no_derived_point_label(tmp_path):
    parent_dir = tmp_path / "parent"
    output_dir = tmp_path / "spatial"
    wiki_path = tmp_path / "wiki.json"
    corine_path = tmp_path / "corine.geojson"

    _make_parent_experiment(parent_dir)
    _write_articles(wiki_path)
    _write_corine_missing_parent_target(corine_path)

    with pytest.raises(ValueError, match="Missing derived full-CORINE point labels"):
        main(
            [
                "--parent-experiment-dir",
                str(parent_dir),
                "--wiki-articles-path",
                str(wiki_path),
                "--corine-polygons-path",
                str(corine_path),
                "--output-dir",
                str(output_dir),
                "--radii",
                "250",
            ]
        )
