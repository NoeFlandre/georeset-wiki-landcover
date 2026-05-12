"""Fetch OSM polygons for CORINE bounds and compare class distributions."""

import argparse
import logging
import os

import geopandas as gpd
import pandas as pd

from georeset.analysis.corine_polygon_stats import corine_distribution_in_osm_polygons
from georeset.cli.data.filter_pipeline import filter_osm_by_corine
from georeset.config import DataPaths
from georeset.fetchers.data_fetcher import DataFetcher
from georeset.fetchers.osm_fetcher import LANDUSE_VALUES, NATURAL_VALUES, OSMFetcher
from georeset.utils.json_io import (
    read_json_file,
    write_csv_atomic,
    write_geojson_atomic,
    write_html_map_atomic,
)
from georeset.visualization.map_visualizer import MapVisualizer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_paths = DataPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-map-path", default=data_paths.map_osm)
    parser.add_argument("--output-csv-path", default=data_paths.distribution_csv)
    parser.add_argument("--osm-polygons-path", default=data_paths.osm_polygons)
    parser.add_argument("--corine-bounds-path", default=data_paths.corine_bounds)
    parser.add_argument(
        "--no-refetch-osm",
        dest="refetch_osm",
        action="store_false",
        help="Load existing OSM polygons instead of fetching fresh Overpass data.",
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args(argv)


def run(
    output_map_path: str = DataPaths().map_osm,
    output_csv_path: str = DataPaths().distribution_csv,
    osm_polygons_path: str = DataPaths().osm_polygons,
    corine_bounds_path: str = DataPaths().corine_bounds,
    refetch_osm: bool = True,
    chunk_size: int = 5000,
) -> None:
    fetcher = DataFetcher()
    corine = fetcher.load_data(exclude_artificial=True)
    if refetch_osm:
        bounds = read_json_file(corine_bounds_path)
        osm = OSMFetcher().fetch_polygons(
            bounds["min_lon"],
            bounds["min_lat"],
            bounds["max_lon"],
            bounds["max_lat"],
        )
    else:
        osm = gpd.read_file(osm_polygons_path)
    osm = osm[osm["landuse"].isin(LANDUSE_VALUES) | osm["natural"].isin(NATURAL_VALUES)].copy()
    osm = filter_osm_by_corine(osm, corine, chunk_size=chunk_size)

    parts = [
        corine_distribution_in_osm_polygons(osm.iloc[start : start + chunk_size], corine)
        for start in range(0, len(osm), chunk_size)
    ]
    distribution = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])
    )

    # Keep only OSM polygons that have at least one CORINE class
    osm = osm[osm["osm_id"].isin(distribution["osm_id"].unique())].copy()

    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(osm_polygons_path), exist_ok=True)

    map_osm = osm.copy()
    map_osm_metric = map_osm.to_crs("EPSG:2154")
    map_osm = map_osm[map_osm_metric.geometry.area >= 15000].copy()

    write_html_map_atomic(
        output_map_path, MapVisualizer(corine).plot_corine_with_osm_polygons(map_osm)
    )
    write_geojson_atomic(osm_polygons_path, osm)
    write_csv_atomic(output_csv_path, distribution, index=False)

    logger.info("Filtered to %s OSM polygons with CORINE classes", len(osm))
    logger.info("Saved map to %s", output_map_path)
    logger.info("Saved distribution to %s", output_csv_path)


def main(argv: list[str] | None = None) -> None:
    """Run the CORINE/OSM analysis CLI."""
    args = parse_args(argv)
    run(
        output_map_path=args.output_map_path,
        output_csv_path=args.output_csv_path,
        osm_polygons_path=args.osm_polygons_path,
        corine_bounds_path=args.corine_bounds_path,
        refetch_osm=args.refetch_osm,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
