"""CORINE buffer-purity diagnostics for article point labels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry.base import BaseGeometry

METRIC_CRS = "EPSG:2154"


def compute_shannon_entropy(shares: Mapping[str, float]) -> float:
    total = sum(value for value in shares.values() if value > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in shares.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log(p)
    return entropy


def compute_normalized_entropy(shares: Mapping[str, float]) -> float:
    n_labels = sum(1 for value in shares.values() if value > 0)
    if n_labels <= 1:
        return 0.0
    return compute_shannon_entropy(shares) / math.log(n_labels)


def derive_level2_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text[:2] if len(text) >= 2 else None


def prepare_corine_gdf(corine_gdf: gpd.GeoDataFrame, label_col: str = "label") -> gpd.GeoDataFrame:
    corine = corine_gdf.copy()
    if corine.crs is None:
        corine = corine.set_crs("EPSG:4326")
    corine = corine.to_crs(METRIC_CRS)
    if label_col not in corine.columns:
        if "code_18" not in corine.columns:
            raise ValueError("CORINE GeoDataFrame must contain either label_col or code_18")
        corine[label_col] = corine["code_18"].map(derive_level2_label)
    corine = corine[corine[label_col].notna()].copy()
    corine["geometry"] = corine.geometry.map(make_valid)
    corine = corine[~corine.geometry.is_empty & corine.geometry.notna()].copy()
    return corine


def _query_candidates(buffer_geom: BaseGeometry, corine_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if corine_gdf.empty:
        return corine_gdf
    indices = corine_gdf.sindex.query(buffer_geom, predicate="intersects")
    return corine_gdf.iloc[indices]


def compute_buffer_label_shares(
    buffer_geom: BaseGeometry, corine_gdf: gpd.GeoDataFrame, label_col: str
) -> dict[str, float]:
    if label_col not in corine_gdf.columns:
        corine_gdf = prepare_corine_gdf(corine_gdf, label_col=label_col)
    if buffer_geom.is_empty or buffer_geom.area <= 0:
        return {}
    buffer_geom = make_valid(buffer_geom)
    buffer_area = buffer_geom.area
    shares: dict[str, float] = {}
    for _, row in _query_candidates(buffer_geom, corine_gdf).iterrows():
        intersection = make_valid(row.geometry).intersection(buffer_geom)
        if intersection.is_empty:
            continue
        area = intersection.area
        if area <= 0:
            continue
        label = str(row[label_col])
        shares[label] = shares.get(label, 0.0) + area / buffer_area
    return dict(sorted(shares.items()))


def _dominant_label(shares: Mapping[str, float]) -> str | None:
    if not shares:
        return None
    return max(sorted(shares), key=lambda label: shares[label])


def _radius_metrics(shares: Mapping[str, float], point_label: str) -> dict[str, Any]:
    dominant_label = _dominant_label(shares)
    dominant_share = shares.get(dominant_label, 0.0) if dominant_label is not None else 0.0
    point_label_share = shares.get(point_label, 0.0)
    coverage_share = min(sum(shares.values()), 1.0)
    return {
        "dominant_label": dominant_label,
        "dominant_share": dominant_share,
        "point_label_share": point_label_share,
        "dominant_matches_point_label": dominant_label == point_label if dominant_label else False,
        "num_labels": len(shares),
        "entropy": compute_shannon_entropy(shares),
        "normalized_entropy": compute_normalized_entropy(shares),
        "coverage_share": coverage_share,
        "label_shares": dict(shares),
    }


def compute_article_spatial_confidence(
    articles_gdf: gpd.GeoDataFrame,
    corine_gdf: gpd.GeoDataFrame,
    radii_m: Sequence[int],
    point_label_col: str,
    pageid_col: str,
) -> pd.DataFrame:
    if articles_gdf.empty:
        return pd.DataFrame()
    articles = articles_gdf.copy()
    if articles.crs is None:
        articles = articles.set_crs("EPSG:4326")
    articles = articles.to_crs(METRIC_CRS)
    articles["geometry"] = articles.geometry.map(make_valid)
    corine = prepare_corine_gdf(corine_gdf, label_col="label")

    rows: list[dict[str, Any]] = []
    for _, article in articles.iterrows():
        point_label = str(article[point_label_col])
        out: dict[str, Any] = {
            "pageid": str(article[pageid_col]),
            "point_label": point_label,
        }
        for radius in radii_m:
            shares = compute_buffer_label_shares(article.geometry.buffer(radius), corine, "label")
            metrics = _radius_metrics(shares, point_label)
            for name, value in metrics.items():
                out[f"{name}_{radius}m"] = value
        rows.append(out)
    result = pd.DataFrame(rows)
    for column in result.columns:
        if column.startswith("dominant_matches_point_label_"):
            result[column] = result[column].astype(object)
    return result
