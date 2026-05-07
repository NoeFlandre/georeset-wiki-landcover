"""Filter pipeline to remove artificial surfaces from all data artifacts.

This script cascades the artificial surfaces exclusion across all data files:
1. Filter CORINE polygons (in memory, no file written)
2. Remove orphan OSM polygons (no intersection with remaining CORINE)
3. Filter wiki_articles.json (remove articles outside filtered CORINE)
4. Filter article_contents.json (keep only pageids in filtered wiki_articles)
5. Filter article_summaries.json (keep only pageids in filtered wiki_articles)
6. Re-run corine analysis to regenerate maps

No API calls are made - all filtering is done locally on existing data.
"""

import argparse
import json
import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.analysis.corine_polygon_stats import corine_distribution_in_osm_polygons
from src.fetchers.data_fetcher import DataFetcher


def filter_articles_by_corine(articles: list[dict], corine_gdf: gpd.GeoDataFrame) -> list[dict]:
    """Keep only articles whose coordinates fall inside at least one CORINE polygon."""
    if not articles or corine_gdf.empty:
        return []

    valid_articles = []
    points = []
    for article in articles:
        lat = article.get("lat")
        lon = article.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            valid_articles.append(article)
            points.append(Point(lon, lat))

    if not valid_articles:
        return []

    points_gdf = gpd.GeoDataFrame(
        {"article_idx": range(len(valid_articles))},
        geometry=points,
        crs=corine_gdf.crs if corine_gdf.crs else "EPSG:4326"
    )

    # Use spatial join for fast point-in-polygon checks
    intersecting = gpd.sjoin(points_gdf, corine_gdf, how="inner", predicate="within")

    # Get unique valid indices
    valid_indices = set(intersecting["article_idx"])

    return [valid_articles[i] for i in valid_indices]


def filter_osm_by_corine(
    osm_gdf: gpd.GeoDataFrame, corine_gdf: gpd.GeoDataFrame, chunk_size: int = 5000
) -> gpd.GeoDataFrame:
    """Remove OSM polygons that have no intersection with CORINE polygons."""
    if osm_gdf.empty or corine_gdf.empty:
        return osm_gdf.iloc[:0].copy()

    # Compute distribution to find which OSM polygons intersect CORINE
    parts = [
        corine_distribution_in_osm_polygons(
            osm_gdf.iloc[start : start + chunk_size],
            corine_gdf,
        )
        for start in range(0, len(osm_gdf), chunk_size)
    ]
    distribution = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])

    valid_osm_ids = set(distribution["osm_id"].unique())
    filtered_osm = osm_gdf[osm_gdf["osm_id"].isin(valid_osm_ids)].copy()
    return filtered_osm


def filter_json_by_pageids(input_path: str, output_path: str, valid_pageids: set[str]) -> None:
    """Filter a JSON file to only keep entries whose string keys are in valid_pageids."""
    if not os.path.exists(input_path):
        print(f"  {input_path}: file not found, skipping")
        return

    with open(input_path) as f:
        data = json.load(f)

    filtered = {k: v for k, v in data.items() if str(k) in valid_pageids}

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"  Filtered {output_path}: {len(data)} -> {len(filtered)} entries")


def filter_articles_json(input_path: str, output_path: str, valid_pageids: set[str]) -> None:
    """Filter wiki_articles.json (list format) to only keep articles with valid pageids."""
    if not os.path.exists(input_path):
        print(f"  {input_path}: file not found, skipping")
        return

    with open(input_path) as f:
        articles = json.load(f)

    valid_str_pageids = {str(pid) for pid in valid_pageids}
    filtered = [a for a in articles if str(a.get("pageid", "")) in valid_str_pageids]

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"  Filtered {output_path}: {len(articles)} -> {len(filtered)} articles")


def regenerate_maps(corine_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame, output_map_path: str) -> None:
    """Re-run map generation with filtered data."""
    from src.visualization.map_visualizer import MapVisualizer

    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)

    # Filter OSM to >= 15000 sqm like run_corine_analysis does
    osm_metric = osm_gdf.to_crs("EPSG:2154")
    osm_filtered = osm_gdf[osm_metric.geometry.area >= 15000].copy()

    map_obj = MapVisualizer(corine_gdf)
    map_obj.plot_corine_with_osm_polygons(osm_filtered).save(output_map_path)
    print(f"  Regenerated map: {output_map_path}")


def regenerate_distribution(
    osm_gdf: gpd.GeoDataFrame, corine_gdf: gpd.GeoDataFrame, output_csv_path: str, chunk_size: int = 5000
) -> None:
    """Re-run CORINE distribution computation with filtered data."""
    parts = [
        corine_distribution_in_osm_polygons(
            osm_gdf.iloc[start : start + chunk_size],
            corine_gdf,
        )
        for start in range(0, len(osm_gdf), chunk_size)
    ]
    distribution = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    distribution.to_csv(output_csv_path, index=False)
    print(f"  Regenerated distribution: {output_csv_path}")


def filter_pipeline(
    wiki_articles_path: str,
    article_contents_path: str,
    article_summaries_path: str,
    osm_polygons_path: str,
    distribution_csv_path: str,
    map_html_path: str,
    output_dir: str | None = None,
) -> None:
    """Main filter pipeline - cascade artificial surfaces exclusion across all artifacts."""
    print("=== Filter Pipeline: Excluding Artificial Surfaces ===\n")

    # Step 1: Load and filter CORINE
    print("Step 1: Loading CORINE data with artificial surfaces excluded...")
    fetcher = DataFetcher()
    corine_gdf = fetcher.load_data(exclude_artificial=True)
    print(f"  CORINE polygons: {len(corine_gdf)} (artificial surfaces excluded)")

    # Step 2: Load and filter OSM polygons
    print("\nStep 2: Removing orphan OSM polygons (no CORINE intersection)...")
    if not os.path.exists(osm_polygons_path):
        print(f"  {osm_polygons_path}: file not found, skipping OSM filter")
        osm_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        osm_gdf = gpd.read_file(osm_polygons_path)
        osm_filtered = filter_osm_by_corine(osm_gdf, corine_gdf)
        osm_gdf = osm_filtered
        osm_gdf.to_file(osm_polygons_path, driver="GeoJSON")
        print(f"  OSM polygons: {len(osm_gdf)} (orphans removed)")

    # Step 3: Filter wiki_articles.json
    print("\nStep 3: Filtering wiki_articles.json...")
    filtered_articles = []
    if not os.path.exists(wiki_articles_path):
        print(f"  {wiki_articles_path}: file not found, skipping")
    else:
        with open(wiki_articles_path) as f:
            articles = json.load(f)
        filtered_articles = filter_articles_by_corine(articles, corine_gdf)
        with open(wiki_articles_path, "w") as f:
            json.dump(filtered_articles, f, indent=2)
        print(f"  wiki_articles.json: {len(articles)} -> {len(filtered_articles)} articles")

    # Step 4: Filter article_contents.json by pageids
    print("\nStep 4: Filtering article_contents.json...")
    valid_pageids = {str(a["pageid"]) for a in filtered_articles}
    filter_json_by_pageids(article_contents_path, article_contents_path, valid_pageids)

    # Step 5: Filter article_summaries.json by pageids
    print("\nStep 5: Filtering article_summaries.json...")
    filter_json_by_pageids(article_summaries_path, article_summaries_path, valid_pageids)

    # Step 6: Regenerate distribution and maps
    print("\nStep 6: Regenerating distribution and maps...")
    if not osm_gdf.empty and len(corine_gdf) > 0:
        regenerate_distribution(osm_gdf, corine_gdf, distribution_csv_path)
        regenerate_maps(corine_gdf, osm_gdf, map_html_path)
    else:
        print("  Skipped: no OSM or CORINE polygons remaining")

    print("\n=== Filter Pipeline Complete ===")


def parse_args():
    parser = argparse.ArgumentParser(description="Filter pipeline to exclude artificial surfaces from all data artifacts.")
    parser.add_argument("--wiki-articles-path", default="data/wiki/wiki_articles.json")
    parser.add_argument("--article-contents-path", default="data/wiki/article_contents.json")
    parser.add_argument("--article-summaries-path", default="data/wiki/article_summaries.json")
    parser.add_argument("--osm-polygons-path", default="data/osm/osm_project_polygons.geojson")
    parser.add_argument("--distribution-csv-path", default="data/distribution/osm_corine_distribution.csv")
    parser.add_argument("--map-html-path", default="data/maps/osm_corine_polygons.html")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_pipeline(
        wiki_articles_path=args.wiki_articles_path,
        article_contents_path=args.article_contents_path,
        article_summaries_path=args.article_summaries_path,
        osm_polygons_path=args.osm_polygons_path,
        distribution_csv_path=args.distribution_csv_path,
        map_html_path=args.map_html_path,
    )
