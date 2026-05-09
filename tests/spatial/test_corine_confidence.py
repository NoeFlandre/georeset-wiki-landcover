import math

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from src.spatial.corine_confidence import (
    compute_article_spatial_confidence,
    compute_buffer_label_shares,
    compute_normalized_entropy,
    compute_shannon_entropy,
)


def _corine(labels, geometries):
    return gpd.GeoDataFrame(
        {"code_18": labels, "geometry": geometries},
        crs="EPSG:2154",
    )


def test_entropy_helpers_handle_pure_and_mixed_shares():
    assert compute_shannon_entropy({"31": 1.0}) == 0.0
    assert compute_normalized_entropy({"31": 1.0}) == 0.0

    entropy = compute_shannon_entropy({"31": 0.5, "21": 0.5})
    assert entropy == pytest.approx(math.log(2))
    assert compute_normalized_entropy({"31": 0.5, "21": 0.5}) == pytest.approx(1.0)


def test_compute_buffer_label_shares_is_area_weighted_and_keeps_artificial_classes():
    corine = _corine(
        ["311", "112"],
        [box(-10, -10, 0, 10), box(0, -10, 10, 10)],
    )
    shares = compute_buffer_label_shares(box(-10, -10, 10, 10), corine, "label")

    assert shares == {"11": 0.5, "31": 0.5}


def test_compute_article_confidence_pure_buffer_has_point_label_share_one():
    articles = gpd.GeoDataFrame(
        {"pageid": ["1"], "point_label": ["31"]},
        geometry=[Point(0, 0)],
        crs="EPSG:2154",
    )
    corine = _corine(["311"], [box(-1000, -1000, 1000, 1000)])

    result = compute_article_spatial_confidence(articles, corine, [250], "point_label", "pageid")
    row = result.iloc[0]

    assert row["point_label_share_250m"] == pytest.approx(1.0)
    assert row["dominant_label_250m"] == "31"
    assert row["dominant_matches_point_label_250m"] is True
    assert row["num_labels_250m"] == 1
    assert row["entropy_250m"] == 0.0
    assert row["normalized_entropy_250m"] == 0.0
    assert row["coverage_share_250m"] == pytest.approx(1.0)


def test_compute_article_confidence_boundary_buffer_can_have_different_dominant_label():
    articles = gpd.GeoDataFrame(
        {"pageid": ["1"], "point_label": ["31"]},
        geometry=[Point(0, 0)],
        crs="EPSG:2154",
    )
    corine = _corine(
        ["311", "211"],
        [box(-1000, -1000, 0, 1000), box(0, -1000, 1000, 1000)],
    )

    result = compute_article_spatial_confidence(articles, corine, [250], "point_label", "pageid")
    row = result.iloc[0]

    assert row["point_label_share_250m"] == pytest.approx(0.5, abs=0.01)
    assert row["dominant_share_250m"] == pytest.approx(0.5, abs=0.01)
    assert row["num_labels_250m"] == 2
    assert row["entropy_250m"] > 0.0


def test_point_label_share_is_zero_when_absent_from_buffer():
    articles = gpd.GeoDataFrame(
        {"pageid": ["1"], "point_label": ["31"]},
        geometry=[Point(0, 0)],
        crs="EPSG:2154",
    )
    corine = _corine(["211"], [box(-1000, -1000, 1000, 1000)])

    result = compute_article_spatial_confidence(articles, corine, [250], "point_label", "pageid")

    assert result.iloc[0]["point_label_share_250m"] == 0.0
    assert result.iloc[0]["dominant_matches_point_label_250m"] is False


def test_no_intersection_buffer_returns_safe_values():
    articles = gpd.GeoDataFrame(
        {"pageid": ["1"], "point_label": ["31"]},
        geometry=[Point(0, 0)],
        crs="EPSG:2154",
    )
    corine = _corine(["211"], [box(10000, 10000, 11000, 11000)])

    result = compute_article_spatial_confidence(articles, corine, [250], "point_label", "pageid")
    row = result.iloc[0]

    assert row["dominant_label_250m"] is None
    assert row["dominant_share_250m"] == 0.0
    assert row["point_label_share_250m"] == 0.0
    assert row["num_labels_250m"] == 0
    assert row["entropy_250m"] == 0.0
    assert row["normalized_entropy_250m"] == 0.0
    assert row["coverage_share_250m"] == 0.0
    assert row["label_shares_250m"] == {}
