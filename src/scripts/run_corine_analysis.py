"""Fetch OSM polygons for CORINE bounds and compare class distributions."""

import json
import os

import geopandas as gpd
import pandas as pd

from src.analysis.corine_polygon_stats import corine_distribution_in_osm_polygons
from src.fetchers.data_fetcher import DataFetcher
from src.visualization.map_visualizer import MapVisualizer
from src.fetchers.osm_fetcher import LANDUSE_VALUES, NATURAL_VALUES, OSMFetcher


def run(
    output_map_path: str = "data/maps/osm_corine_polygons.html",
    output_csv_path: str = "data/distribution/osm_corine_distribution.csv",
    output_osm_path: str = "data/osm/osm_project_polygons.geojson",
    chunk_size: int = 5000,
):
    with open("data/corine/bounds.json") as f:
        bounds = json.load(f)

    corine = DataFetcher().load_data()
    if output_osm_path:
        osm = gpd.read_file(output_osm_path)
    else:
        osm = OSMFetcher().fetch_polygons(
            bounds["min_lon"],
            bounds["min_lat"],
            bounds["max_lon"],
            bounds["max_lat"],
        )
    osm = osm[osm["landuse"].isin(LANDUSE_VALUES) | osm["natural"].isin(NATURAL_VALUES)].copy()

    parts = [
        corine_distribution_in_osm_polygons(osm.iloc[start:start + chunk_size], corine)
        for start in range(0, len(osm), chunk_size)
    ]
    distribution = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])

    # Keep only OSM polygons that have at least one CORINE class
    osm = osm[osm["osm_id"].isin(distribution["osm_id"].unique())].copy()

    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)

    map_osm = osm.copy()
    map_osm_metric = map_osm.to_crs("EPSG:2154")
    map_osm = map_osm[map_osm_metric.geometry.area >= 15000].copy()

    MapVisualizer(corine).plot_corine_with_osm_polygons(map_osm).save(output_map_path)
    osm.to_file(output_osm_path, driver="GeoJSON")
    distribution.to_csv(output_csv_path, index=False)

    print(f"Filtered to {len(osm)} OSM polygons with CORINE classes")
    print(f"Saved map to {output_map_path}")
    print(f"Saved distribution to {output_csv_path}")


if __name__ == "__main__":
    run()
