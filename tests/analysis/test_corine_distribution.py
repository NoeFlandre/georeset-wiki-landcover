"""Tests for CORINE class distribution inside OSM polygons."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from georeset.analysis.corine_polygon_stats import corine_distribution_in_osm_polygons


def test_distribution_reports_area_share_per_osm_polygon():
    osm = gpd.GeoDataFrame(
        {"osm_id": [1], "geometry": [box(0, 0, 10, 10)]},
        crs="EPSG:3857",
    )
    corine = gpd.GeoDataFrame(
        {
            "code_18": ["311", "112"],
            "geometry": [box(0, 0, 4, 10), box(4, 0, 10, 10)],
        },
        crs="EPSG:3857",
    )

    distribution = corine_distribution_in_osm_polygons(osm, corine, metric_crs="EPSG:3857")

    assert list(distribution.columns) == ["osm_id", "class_label", "area", "share"]
    assert distribution.to_dict("records") == [
        {"osm_id": 1, "class_label": "112", "area": 60.0, "share": 0.6},
        {"osm_id": 1, "class_label": "311", "area": 40.0, "share": 0.4},
    ]


def test_distribution_aggregates_same_class_fragments():
    osm = gpd.GeoDataFrame(
        {"osm_id": [1], "geometry": [box(0, 0, 10, 10)]},
        crs="EPSG:3857",
    )
    corine = gpd.GeoDataFrame(
        {
            "code_18": ["311", "311"],
            "geometry": [box(0, 0, 2, 10), box(2, 0, 5, 10)],
        },
        crs="EPSG:3857",
    )

    distribution = corine_distribution_in_osm_polygons(osm, corine, metric_crs="EPSG:3857")

    assert distribution.to_dict("records") == [
        {"osm_id": 1, "class_label": "311", "area": 50.0, "share": 1.0},
    ]


def test_distribution_can_process_osm_polygons_in_chunks():
    osm = gpd.GeoDataFrame(
        {"osm_id": [1, 2], "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)]},
        crs="EPSG:3857",
    )
    corine = gpd.GeoDataFrame(
        {
            "code_18": ["311", "112"],
            "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)],
        },
        crs="EPSG:3857",
    )

    distribution = corine_distribution_in_osm_polygons(
        osm, corine, metric_crs="EPSG:3857", chunk_size=1
    )

    assert distribution.to_dict("records") == [
        {"osm_id": 1, "class_label": "311", "area": 100.0, "share": 1.0},
        {"osm_id": 2, "class_label": "112", "area": 100.0, "share": 1.0},
    ]


def test_distribution_requires_identifier_column():
    osm = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs="EPSG:3857")
    corine = gpd.GeoDataFrame(
        {"code_18": ["311"], "geometry": [box(0, 0, 10, 10)]},
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="osm_id"):
        corine_distribution_in_osm_polygons(osm, corine)
