"""Compute CORINE class distributions inside OSM polygons."""

import geopandas as gpd
import pandas as pd


def corine_distribution_in_osm_polygons(
    osm_polygons: gpd.GeoDataFrame,
    corine_polygons: gpd.GeoDataFrame,
    osm_id_col: str = "osm_id",
    corine_class_col: str = "code_18",
    metric_crs: str = "EPSG:2154",
    chunk_size: int | None = None,
) -> pd.DataFrame:
    """Return area and share of CORINE classes intersecting each OSM polygon."""
    _require_columns(osm_polygons, [osm_id_col])
    _require_columns(corine_polygons, [corine_class_col])

    if osm_polygons.empty or corine_polygons.empty:
        return _empty_result()

    if chunk_size:
        parts = [
            corine_distribution_in_osm_polygons(
                osm_polygons.iloc[start:start + chunk_size],
                corine_polygons,
                osm_id_col=osm_id_col,
                corine_class_col=corine_class_col,
                metric_crs=metric_crs,
            )
            for start in range(0, len(osm_polygons), chunk_size)
        ]
        if not parts:
            return _empty_result()
        return pd.concat(parts, ignore_index=True).sort_values(["osm_id", "class_label"]).reset_index(drop=True)

    osm = osm_polygons[[osm_id_col, "geometry"]].to_crs(metric_crs)
    corine = corine_polygons[[corine_class_col, "geometry"]].to_crs(metric_crs)

    intersections = gpd.overlay(osm, corine, how="intersection", keep_geom_type=True)
    if intersections.empty:
        return _empty_result()

    intersections["area"] = intersections.geometry.area
    grouped = (
        intersections.groupby([osm_id_col, corine_class_col], as_index=False)["area"]
        .sum()
        .rename(columns={osm_id_col: "osm_id", corine_class_col: "class_label"})
    )
    totals = grouped.groupby("osm_id")["area"].transform("sum")
    grouped["share"] = grouped["area"] / totals
    return grouped.sort_values(["osm_id", "class_label"]).reset_index(drop=True)


def _require_columns(gdf: gpd.GeoDataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required column: {missing[0]}")


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])
