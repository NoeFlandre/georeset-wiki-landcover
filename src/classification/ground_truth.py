import geopandas as gpd
from shapely.geometry import Point

from src.classification.labels import osm_labels_from_row


def _articles_to_points(articles: list[dict]) -> gpd.GeoDataFrame:
    rows, points = [], []
    for article in articles:
        lat, lon, pageid = article.get("lat"), article.get("lon"), article.get("pageid")
        if pageid is None or not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        rows.append({"pageid": str(pageid)})
        points.append(Point(lon, lat))
    return gpd.GeoDataFrame(rows, geometry=points, crs="EPSG:4326")


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def build_corine_ground_truth(
    articles: list[dict], corine_gdf: gpd.GeoDataFrame
) -> dict[str, str]:
    """
    CORINE ground truth: single-label per pageid.
    - Exactly one level-2 label → keep.
    - Zero or multiple distinct level-2 labels → exclude (no dict entry).
    """
    points = _articles_to_points(articles)
    if points.empty or corine_gdf.empty:
        return {}
    corine = _to_wgs84(corine_gdf).copy()
    corine = corine[~corine["code_18"].astype(str).str.startswith("1")].copy()
    corine["label"] = corine["code_18"].astype(str).str[:2]
    joined = gpd.sjoin(
        points, corine[["label", "geometry"]], how="inner", predicate="intersects"
    )
    result = {}
    for pageid, group in joined.groupby("pageid", sort=False):
        distinct = sorted(set(group["label"].astype(str)))
        if len(distinct) == 1:
            result[str(pageid)] = distinct[0]
    return result


def build_osm_ground_truth(
    articles: list[dict], osm_gdf: gpd.GeoDataFrame
) -> dict[str, list[str]]:
    """
    OSM ground truth: multi-label per pageid.
    - Collect all osm_labels_from_row results, deduplicate, sort.
    - Zero allowed labels → exclude (no dict entry).
    """
    points = _articles_to_points(articles)
    if points.empty or osm_gdf.empty:
        return {}
    osm = _to_wgs84(osm_gdf).copy()
    osm["labels"] = osm.apply(osm_labels_from_row, axis=1)
    osm = osm[osm["labels"].astype(bool)].copy()
    if osm.empty:
        return {}
    osm = osm.explode("labels")
    osm = osm.rename(columns={"labels": "label"})
    joined = gpd.sjoin(
        points, osm[["label", "geometry"]], how="inner", predicate="intersects"
    )
    result = {}
    for pageid, group in joined.groupby("pageid", sort=False):
        labels = sorted(set(group["label"].astype(str)))
        if labels:
            result[str(pageid)] = labels
    return result
