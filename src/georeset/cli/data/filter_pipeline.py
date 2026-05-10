"""Filter pipeline to remove artificial surfaces from all data artifacts.

This script cascades the artificial surfaces exclusion across all data files:
1. Filter CORINE polygons (in memory, no file written)
2. Remove orphan OSM polygons (intersection with filtered CORINE only)
3. Filter wiki_articles.json (remove articles outside filtered CORINE OR filtered OSM)
4. Prune article_contents.json to filtered wiki_articles pageids
5. Prune article_summaries.json to filtered article_contents keys
6. Prune both summary variants to final article content keys
7. Regenerate distribution and both maps

Supports --refetch-osm, --refetch-wiki, --fetch-content, --summarize,
--audit-only, and --dry-run.
"""

import argparse
import json
import logging
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from georeset.analysis.corine_polygon_stats import corine_distribution_in_osm_polygons
from georeset.config import DataPaths, ModelSettings
from georeset.contracts import ArticleMeta
from georeset.fetchers.article_summarizer import ArticleSummarizer
from georeset.fetchers.data_fetcher import DataFetcher
from georeset.fetchers.osm_fetcher import OSMFetcher
from georeset.fetchers.wiki_content_fetcher import WikiContentFetcher
from georeset.fetchers.wiki_fetcher import WikiFetcher
from georeset.spatial.policy import POINT_POLYGON_JOIN_PREDICATE
from georeset.utils.json_io import (
    write_csv_atomic,
    write_geojson_atomic,
    write_html_map_atomic,
    write_json_atomic,
)
from georeset.visualization.map_visualizer import MapVisualizer

logger = logging.getLogger(__name__)


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def filter_articles_by_polygons(
    articles: list[ArticleMeta],
    corine_gdf: gpd.GeoDataFrame,
    osm_gdf: gpd.GeoDataFrame,
) -> list[ArticleMeta]:
    """Keep articles intersecting filtered CORINE OR filtered OSM point/polygon boundaries.

    The project uses the same point-polygon predicate for filtering and
    classification ground truth. Boundary points are retained here, then
    downstream ground-truth builders resolve ambiguity by excluding CORINE
    pageids with multiple distinct level-2 labels.
    """
    if not articles:
        return []

    seen_pageids: set[str] = set()
    kept_articles: list[ArticleMeta] = []
    points: list[Point] = []
    kept_positions: list[int] = []

    for article in articles:
        pid = str(article.get("pageid", ""))
        if pid in seen_pageids:
            continue
        lat = article.get("lat")
        lon = article.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        seen_pageids.add(pid)
        kept_positions.append(len(kept_articles))
        kept_articles.append(article)
        points.append(Point(lon, lat))

    if not kept_articles:
        return []

    corine_4326 = _to_wgs84(corine_gdf)
    osm_4326 = _to_wgs84(osm_gdf)

    points_gdf = gpd.GeoDataFrame({"kept_pos": kept_positions}, geometry=points, crs="EPSG:4326")

    in_corine = set()
    in_osm = set()

    if not corine_4326.empty:
        joined = gpd.sjoin(
            points_gdf,
            corine_4326,
            how="inner",
            predicate=POINT_POLYGON_JOIN_PREDICATE,
        )
        in_corine = set(joined["kept_pos"].tolist())

    if not osm_4326.empty:
        joined = gpd.sjoin(
            points_gdf,
            osm_4326,
            how="inner",
            predicate=POINT_POLYGON_JOIN_PREDICATE,
        )
        in_osm = set(joined["kept_pos"].tolist())

    valid_positions = in_corine | in_osm
    return [kept_articles[i] for i in sorted(valid_positions)]


def filter_osm_by_corine(
    osm_gdf: gpd.GeoDataFrame,
    corine_gdf: gpd.GeoDataFrame,
    chunk_size: int = 5000,
) -> gpd.GeoDataFrame:
    """Keep OSM polygons that intersect at least one filtered (non-artificial) CORINE polygon."""
    if osm_gdf.empty or corine_gdf.empty:
        return osm_gdf.iloc[:0].copy()
    distribution = _distribution_in_chunks(osm_gdf, corine_gdf, chunk_size)
    valid_osm_ids = set(distribution["osm_id"].unique())
    return osm_gdf[osm_gdf["osm_id"].isin(valid_osm_ids)].copy()


def _distribution_in_chunks(
    osm_gdf: gpd.GeoDataFrame, corine_gdf: gpd.GeoDataFrame, chunk_size: int
) -> pd.DataFrame:
    parts = [
        corine_distribution_in_osm_polygons(
            osm_gdf.iloc[start : start + chunk_size],
            corine_gdf,
        )
        for start in range(0, len(osm_gdf), chunk_size)
    ]
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])
    )


def _prune_json_file(path: str, valid_keys: set[str]) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    filtered = {k: v for k, v in data.items() if str(k) in valid_keys}
    write_json_atomic(path, filtered, indent=2)


def _validate_inputs_before_write(wiki_articles_path: str, refetch_wiki: bool) -> None:
    if not refetch_wiki and not os.path.exists(wiki_articles_path):
        raise FileNotFoundError(
            f"wiki articles file missing: {wiki_articles_path}. "
            "Use --refetch-wiki or provide an existing wiki_articles.json before filtering."
        )


def regenerate_maps(
    corine_gdf: gpd.GeoDataFrame, osm_gdf: gpd.GeoDataFrame, output_map_path: str
) -> None:
    """Re-run map generation with filtered data."""
    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)

    # Filter OSM to >= 15000 sqm like run_corine_analysis does
    osm_metric = osm_gdf.to_crs("EPSG:2154")
    osm_filtered = osm_gdf[osm_metric.geometry.area >= 15000].copy()

    write_html_map_atomic(
        output_map_path,
        MapVisualizer(corine_gdf).plot_corine_with_osm_polygons(osm_filtered),
    )
    logger.info("Regenerated map: %s", output_map_path)


def regenerate_distribution(
    osm_gdf: gpd.GeoDataFrame,
    corine_gdf: gpd.GeoDataFrame,
    output_csv_path: str,
    chunk_size: int = 5000,
) -> None:
    """Re-run CORINE distribution computation with filtered data."""
    parts = [
        corine_distribution_in_osm_polygons(
            osm_gdf.iloc[start : start + chunk_size],
            corine_gdf,
        )
        for start in range(0, len(osm_gdf), chunk_size)
    ]
    distribution = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["osm_id", "class_label", "area", "share"])
    )

    write_csv_atomic(output_csv_path, distribution, index=False)
    logger.info("Regenerated distribution: %s", output_csv_path)


def audit_artifacts(
    corine_gdf: gpd.GeoDataFrame,
    osm_polygons_path: str,
    wiki_articles_path: str,
    article_contents_path: str,
    article_summaries_path: str,
    article_summaries_no_place_path: str,
    distribution_csv_path: str,
    map_articles_path: str,
    map_osm_path: str,
) -> list[str]:
    """Return list of violations; empty means all checks pass."""
    violations = []

    # 1. CORINE has no artificial classes
    artificial = corine_gdf[corine_gdf["code_18"].str.startswith("1")]
    if len(artificial) > 0:
        violations.append(f"CORINE has {len(artificial)} artificial polygons")

    # 2. Required files exist
    for path in [wiki_articles_path, article_contents_path]:
        if not os.path.exists(path):
            violations.append(f"required file missing: {path}")

    # 3. OSM overlaps filtered CORINE (use provided osm_polygons_path)
    osm = (
        gpd.read_file(osm_polygons_path)
        if os.path.exists(osm_polygons_path)
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    osm_filtered = filter_osm_by_corine(osm, corine_gdf)
    if len(osm_filtered) != len(osm):
        violations.append(
            f"OSM has {len(osm)} total but {len(osm_filtered)} overlap filtered CORINE"
        )

    # 4. Every article is in CORINE or OSM (set-diff logic)
    if os.path.exists(wiki_articles_path):
        with open(wiki_articles_path) as f:
            articles = json.load(f)
        valid_ids = {
            str(a["pageid"])
            for a in filter_articles_by_polygons(articles, corine_gdf, osm_filtered)
        }
        all_ids = {str(a["pageid"]) for a in articles}
        invalid_ids = all_ids - valid_ids
        if invalid_ids:
            violations.append(f"{len(invalid_ids)} articles outside CORINE and OSM: {invalid_ids}")

    # 5. article_contents keys ⊆ wiki_articles pageids
    if os.path.exists(wiki_articles_path) and os.path.exists(article_contents_path):
        with open(wiki_articles_path) as f:
            wiki = json.load(f)
        with open(article_contents_path) as f:
            contents = json.load(f)
        wiki_pageids = {str(a["pageid"]) for a in wiki}
        stray_contents = set(contents.keys()) - wiki_pageids
        if stray_contents:
            violations.append(f"article_contents has stale keys: {stray_contents}")

    # 6. article_summaries keys ⊆ article_contents keys
    summary_paths = [
        ("article_summaries", article_summaries_path),
        ("article_summaries_no_place", article_summaries_no_place_path),
    ]
    for name, summary_path in summary_paths:
        if not os.path.exists(article_contents_path) or not os.path.exists(summary_path):
            continue
        with open(article_contents_path) as f:
            contents = json.load(f)
        with open(summary_path) as f:
            summaries = json.load(f)
        stray_summaries = set(summaries.keys()) - set(contents.keys())
        if stray_summaries:
            violations.append(f"{name} has stale keys: {stray_summaries}")

    # 7. Distribution exists and has no artificial class labels
    if not os.path.exists(distribution_csv_path):
        violations.append(f"distribution missing: {distribution_csv_path}")
    else:
        dist = pd.read_csv(distribution_csv_path)
        artificial_rows = dist[dist["class_label"].astype(str).str.startswith("1")]
        if len(artificial_rows) > 0:
            violations.append(f"distribution has {len(artificial_rows)} artificial class rows")

    # 8. Both maps exist with non-zero size
    for path in [map_articles_path, map_osm_path]:
        if not os.path.exists(path):
            violations.append(f"map missing: {path}")
        elif os.path.getsize(path) == 0:
            violations.append(f"map empty: {path}")

    return violations


def filter_pipeline(
    wiki_articles_path: str = DataPaths().wiki_articles,
    article_contents_path: str = DataPaths().article_contents,
    article_summaries_path: str = DataPaths().article_summaries,
    article_summaries_no_place_path: str = DataPaths().article_summaries_no_place,
    osm_polygons_path: str = DataPaths().osm_polygons,
    distribution_csv_path: str = DataPaths().distribution_csv,
    map_articles_path: str = DataPaths().map_articles,
    map_osm_path: str = DataPaths().map_osm,
    refetch_osm: bool = False,
    refetch_wiki: bool = False,
    fetch_content: bool = False,
    summarize: bool = False,
    model_path: str | None = None,
    seed: int = 42,
    temperature: float = 0.7,
    audit_only: bool = False,
    dry_run: bool = False,
) -> None:
    """Orchestrate the full non-artificial CORINE cascade."""
    _validate_inputs_before_write(wiki_articles_path, refetch_wiki)

    # 1. Load filtered CORINE only (no full_corine)
    corine_gdf = DataFetcher().load_data(exclude_artificial=True)

    # 2. Audit mode — short-circuit before mutation
    if audit_only:
        violations = audit_artifacts(
            corine_gdf,
            osm_polygons_path,
            wiki_articles_path,
            article_contents_path,
            article_summaries_path,
            article_summaries_no_place_path,
            distribution_csv_path,
            map_articles_path,
            map_osm_path,
        )
        if violations:
            logger.error("%s", "\n".join(violations))
            raise SystemExit(1)
        logger.info("Audit passed")
        return

    # 3. Load / refetch OSM, then filter to overlap with filtered CORINE
    if refetch_osm:
        with open(DataPaths().corine_bounds) as f:
            bounds = json.load(f)
        osm_gdf = OSMFetcher().fetch_polygons(
            bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"]
        )
    else:
        osm_gdf = (
            gpd.read_file(osm_polygons_path)
            if os.path.exists(osm_polygons_path)
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )

    osm_gdf = filter_osm_by_corine(osm_gdf, corine_gdf)

    # 4. Filter wiki articles: kept if inside CORINE OR OSM
    filtered_articles = []
    if refetch_wiki:
        with open(DataPaths().corine_bounds) as f:
            bounds = json.load(f)
        # Fetch without polygon filters; trust the post-filter
        all_articles = WikiFetcher().get_articles_in_bounds(
            bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"]
        )
        filtered_articles = filter_articles_by_polygons(all_articles, corine_gdf, osm_gdf)
    elif os.path.exists(wiki_articles_path):
        with open(wiki_articles_path) as f:
            articles = json.load(f)
        filtered_articles = filter_articles_by_polygons(articles, corine_gdf, osm_gdf)

    if dry_run:
        logger.info(
            "Dry run: "
            f"would write {len(osm_gdf)} OSM polygons and "
            f"{len(filtered_articles)} filtered wiki articles; "
            "no files changed."
        )
        return

    # 5. WRITE FILTERED OSM (handle empty case safely)
    os.makedirs(os.path.dirname(osm_polygons_path), exist_ok=True)
    if osm_gdf.empty:
        osm_gdf = gpd.GeoDataFrame(
            [],
            columns=[
                "osm_id",
                "name",
                "landuse",
                "natural",
                "leisure",
                "amenity",
                "building",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
    write_geojson_atomic(osm_polygons_path, osm_gdf)

    write_json_atomic(wiki_articles_path, filtered_articles, indent=2)

    valid_pageids = {str(a["pageid"]) for a in filtered_articles}

    # 6. Prune article_contents.json to valid_pageids
    _prune_json_file(article_contents_path, valid_pageids)

    # 7. Fetch missing content if requested
    if fetch_content:
        WikiContentFetcher().fetch_from_file(wiki_articles_path, article_contents_path)

    # 8. Prune article_summaries.json to final article_contents.json keys
    if os.path.exists(article_contents_path):
        with open(article_contents_path) as f:
            contents = json.load(f)
        content_keys = set(contents.keys())
        _prune_json_file(article_summaries_path, content_keys)
        _prune_json_file(article_summaries_no_place_path, content_keys)

    # 9. Summarize if requested
    if summarize:
        ArticleSummarizer(
            model_path=model_path or ModelSettings().model_path,
            seed=seed,
            temperature=temperature,
        ).process_file(article_contents_path, article_summaries_path)

    # 10. Regenerate distribution
    if not osm_gdf.empty and len(corine_gdf) > 0:
        regenerate_distribution(osm_gdf, corine_gdf, distribution_csv_path)

    # 11. Regenerate both maps
    if not osm_gdf.empty and len(corine_gdf) > 0:
        regenerate_maps(corine_gdf, osm_gdf, map_osm_path)
        os.makedirs(os.path.dirname(map_articles_path), exist_ok=True)
        wiki_articles = []
        if os.path.exists(wiki_articles_path):
            with open(wiki_articles_path) as f:
                wiki_articles = json.load(f)
        write_html_map_atomic(
            map_articles_path,
            MapVisualizer(corine_gdf).plot_polygons_with_articles(wiki_articles),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_paths = DataPaths()
    parser = argparse.ArgumentParser(
        description="Filter pipeline to exclude artificial surfaces from all data artifacts."
    )
    parser.add_argument("--wiki-articles-path", default=data_paths.wiki_articles)
    parser.add_argument("--article-contents-path", default=data_paths.article_contents)
    parser.add_argument("--article-summaries-path", default=data_paths.article_summaries)
    parser.add_argument(
        "--article-summaries-no-place-path", default=data_paths.article_summaries_no_place
    )
    parser.add_argument("--osm-polygons-path", default=data_paths.osm_polygons)
    parser.add_argument("--distribution-csv-path", default=data_paths.distribution_csv)
    parser.add_argument("--map-articles-path", default=data_paths.map_articles)
    parser.add_argument("--map-osm-path", default=data_paths.map_osm)
    parser.add_argument("--refetch-osm", action="store_true")
    parser.add_argument("--refetch-wiki", action="store_true")
    parser.add_argument("--fetch-content", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the artificial-surface filtering pipeline CLI."""
    args = parse_args(argv)
    filter_pipeline(
        wiki_articles_path=args.wiki_articles_path,
        article_contents_path=args.article_contents_path,
        article_summaries_path=args.article_summaries_path,
        article_summaries_no_place_path=args.article_summaries_no_place_path,
        osm_polygons_path=args.osm_polygons_path,
        distribution_csv_path=args.distribution_csv_path,
        map_articles_path=args.map_articles_path,
        map_osm_path=args.map_osm_path,
        refetch_osm=args.refetch_osm,
        refetch_wiki=args.refetch_wiki,
        fetch_content=args.fetch_content,
        summarize=args.summarize,
        model_path=args.model_path,
        seed=args.seed,
        temperature=args.temperature,
        audit_only=args.audit_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
