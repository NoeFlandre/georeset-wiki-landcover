from unittest.mock import MagicMock, patch

import geopandas as gpd
from shapely.geometry import box

from src.classification.labels import CORINE_LEVEL2_DESCRIPTIONS
from src.classification.task_setup import ClassificationTaskSetup, load_task_setup


def test_load_corine_task_setup_builds_target_labels_and_descriptions():
    articles = [{"pageid": 100, "lat": 0.5, "lon": 0.5}]
    corine = gpd.GeoDataFrame(
        {"code_18": ["311"], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:4326",
    )
    fetcher = MagicMock()
    fetcher.load_data.return_value = corine

    with patch("src.classification.task_setup.DataFetcher", return_value=fetcher):
        setup = load_task_setup(
            task="corine_level2",
            articles=articles,
            corine_polygons_path="corine.shp",
            osm_polygons_path="osm.geojson",
        )

    assert setup == ClassificationTaskSetup(
        target={"100": "31"},
        allowed_labels=["31"],
        label_descriptions={"31": CORINE_LEVEL2_DESCRIPTIONS["31"]},
    )
    fetcher.load_data.assert_called_once_with(exclude_artificial=True)


def test_load_osm_task_setup_builds_multilabel_target_without_descriptions(tmp_path):
    articles = [{"pageid": 100, "lat": 0.5, "lon": 0.5}]
    osm_path = tmp_path / "osm.geojson"
    osm = gpd.GeoDataFrame(
        {"osm_id": ["way/1"], "landuse": ["meadow"], "natural": ["wood"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    osm.to_file(osm_path, driver="GeoJSON")

    setup = load_task_setup(
        task="osm",
        articles=articles,
        corine_polygons_path="corine.shp",
        osm_polygons_path=str(osm_path),
    )

    assert setup.target == {"100": ["meadow", "wood"]}
    assert "meadow" in setup.allowed_labels
    assert "wood" in setup.allowed_labels
    assert setup.label_descriptions == {}


def test_load_task_setup_rejects_unknown_task():
    try:
        load_task_setup(
            task="unknown",
            articles=[],
            corine_polygons_path="corine.shp",
            osm_polygons_path="osm.geojson",
        )
    except ValueError as exc:
        assert "Unknown classification task" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
