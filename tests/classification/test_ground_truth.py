import geopandas as gpd
from shapely.geometry import box

from src.classification.ground_truth import build_corine_ground_truth, build_osm_ground_truth


def test_build_corine_ground_truth_uses_level2_label():
    articles = [
        {"pageid": 100, "lat": 0.5, "lon": 0.5},
        {"pageid": 200, "lat": 2.5, "lon": 2.5},
    ]
    corine = gpd.GeoDataFrame(
        {"code_18": ["211", "311"], "geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3)]},
        crs="EPSG:4326",
    )
    assert build_corine_ground_truth(articles, corine) == {"100": "21", "200": "31"}


def test_build_corine_ground_truth_excludes_pageids_with_multiple_labels():
    articles = [{"pageid": 100, "lat": 0.5, "lon": 0.5}]
    corine = gpd.GeoDataFrame(
        {"code_18": ["211", "311"], "geometry": [box(0, 0, 2, 2), box(0, 0, 2, 2)]},
        crs="EPSG:4326",
    )
    assert build_corine_ground_truth(articles, corine) == {}


def test_build_osm_ground_truth_is_multilabel_and_ignores_out_of_scope():
    articles = [{"pageid": 100, "lat": 0.5, "lon": 0.5}]
    osm = gpd.GeoDataFrame(
        {
            "osm_id": ["way/1", "way/2", "way/3"],
            "landuse": ["meadow", None, "residential"],
            "natural": [None, "wood", None],
            "geometry": [box(0, 0, 1, 1)] * 3,
        },
        crs="EPSG:4326",
    )
    assert build_osm_ground_truth(articles, osm) == {"100": ["meadow", "wood"]}


def test_build_osm_ground_truth_excludes_articles_without_allowed_osm_labels():
    articles = [{"pageid": 100, "lat": 5.0, "lon": 5.0}]
    osm = gpd.GeoDataFrame(
        {
            "osm_id": ["way/1"],
            "landuse": ["meadow"],
            "natural": [None],
            "geometry": [box(0, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )
    assert build_osm_ground_truth(articles, osm) == {}


def test_build_osm_ground_truth_keeps_landuse_and_natural_on_same_polygon():
    articles = [{"pageid": 100, "lat": 0.5, "lon": 0.5}]
    osm = gpd.GeoDataFrame(
        {
            "osm_id": ["way/1"],
            "landuse": ["meadow"],
            "natural": ["wood"],
            "geometry": [box(0, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )
    assert build_osm_ground_truth(articles, osm) == {"100": ["meadow", "wood"]}


def test_build_corine_ground_truth_boundary_same_level2_is_kept():
    articles = [{"pageid": 100, "lat": 0.5, "lon": 1.0}]
    corine = gpd.GeoDataFrame(
        {"code_18": ["311", "312"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
        crs="EPSG:4326",
    )
    assert build_corine_ground_truth(articles, corine) == {"100": "31"}


def test_build_corine_ground_truth_boundary_different_level2_is_excluded():
    articles = [{"pageid": 100, "lat": 0.5, "lon": 1.0}]
    corine = gpd.GeoDataFrame(
        {"code_18": ["211", "311"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
        crs="EPSG:4326",
    )
    assert build_corine_ground_truth(articles, corine) == {}
