import geopandas as gpd
from shapely.geometry import box

from georeset_wiki_landcover.classification.labels import (
    corine_level2_labels,
    osm_allowed_labels,
    osm_label_from_row,
    osm_labels_from_row,
)
from georeset_wiki_landcover.fetchers.osm_fetcher import LANDUSE_VALUES, NATURAL_VALUES


def test_corine_level2_labels_derive_first_two_digits_and_drop_artificial():
    gdf = gpd.GeoDataFrame(
        {"code_18": ["211", "221", "311", "512", "112"], "geometry": [box(0, 0, 1, 1)] * 5},
        crs="EPSG:4326",
    )
    assert corine_level2_labels(gdf) == ["21", "22", "31", "51"]


def test_osm_label_prefers_allowed_landuse_over_natural():
    row = {"landuse": "meadow", "natural": "wood"}
    assert osm_label_from_row(row) == "meadow"


def test_osm_label_uses_allowed_natural_when_landuse_missing():
    row = {"landuse": None, "natural": "wood"}
    assert osm_label_from_row(row) == "wood"


def test_osm_label_ignores_out_of_scope_tags():
    row = {"landuse": "residential", "natural": "tree_row"}
    assert osm_label_from_row(row) is None


def test_osm_allowed_labels_are_project_scope_constants():
    assert osm_allowed_labels() == sorted(set(LANDUSE_VALUES) | set(NATURAL_VALUES))


def test_osm_labels_from_row_returns_landuse_and_natural_when_both_allowed():
    row = {"landuse": "meadow", "natural": "wood"}
    assert osm_labels_from_row(row) == ["meadow", "wood"]
