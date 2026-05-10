"""OSM map visualizer for Corine Land Cover polygons."""

import logging
from typing import Any

import folium
import geopandas as gpd

from georeset.config import DataPaths

DEFAULT_MAP_LOCATION = [48.5, 7.5]
logger = logging.getLogger(__name__)


class MapVisualizer:
    """Visualizes GeoDataFrame polygons on an OpenStreetMap."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf

    def _map_for_bounds(self, zoom_start: int = 8, *gdfs: gpd.GeoDataFrame) -> folium.Map:
        non_empty = [gdf for gdf in (self.gdf, *gdfs) if not gdf.empty]
        if not non_empty:
            return folium.Map(location=DEFAULT_MAP_LOCATION, zoom_start=zoom_start)

        bounds = non_empty[0].total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        return folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    def plot_polygons(self, zoom_start: int = 8) -> folium.Map:
        """Plot polygons on an OSM map. Returns a folium Map."""
        m = self._map_for_bounds(zoom_start)

        for _, row in self.gdf.iterrows():
            folium.GeoJson(row["geometry"]).add_to(m)

        return m

    def plot_polygons_with_articles(
        self, articles: list[dict[str, Any]], zoom_start: int = 8
    ) -> folium.Map:
        """Plot polygons and Wikipedia article locations as dots."""
        m = self._map_for_bounds(zoom_start)

        for _, row in self.gdf.iterrows():
            folium.GeoJson(row["geometry"]).add_to(m)

        for article in articles:
            folium.CircleMarker(
                location=[article["lat"], article["lon"]],
                radius=4,
                color="red",
                fill=True,
                fillColor="red",
                fillOpacity=0.7,
                popup=article.get("title", "Article"),
            ).add_to(m)

        legend_html = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border: 2px solid gray;
                    font-size: 14px; border-radius: 5px;">
            <b>Legend</b><br>
            <i style="background-color: blue; width: 20px; height: 10px; display: inline-block;"></i> Polygons: {len(self.gdf)}<br>
            <i style="background-color: red; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Articles: {len(articles)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]

        return m

    def plot_corine_with_osm_polygons(
        self, osm_gdf: gpd.GeoDataFrame, zoom_start: int = 8
    ) -> folium.Map:
        """Plot CORINE and OSM polygons as separate layers for visual checking."""
        m = self._map_for_bounds(zoom_start, osm_gdf)

        folium.GeoJson(
            self.gdf,
            name="CORINE polygons",
            style_function=lambda _: {"color": "blue", "weight": 1, "fillOpacity": 0.15},
        ).add_to(m)
        folium.GeoJson(
            osm_gdf,
            name="OSM polygons",
            style_function=lambda _: {"color": "red", "weight": 2, "fillOpacity": 0.05},
        ).add_to(m)
        folium.LayerControl().add_to(m)

        legend_html = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background-color: white; padding: 10px; border: 2px solid gray;
                    font-size: 12px; border-radius: 5px;">
            <b>Legend</b><br>
            <i style="background-color: blue; width: 20px; height: 10px; display: inline-block;"></i> CORINE: {len(self.gdf)}<br>
            <i style="background-color: red; width: 20px; height: 10px; display: inline-block;"></i> OSM: {len(osm_gdf)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]

        return m

    def save_map(self, output_path: str, articles: list[dict[str, Any]] | None = None) -> None:
        """Save the map as an HTML file."""
        m = self.plot_polygons_with_articles(articles) if articles else self.plot_polygons()
        m.save(output_path)


if __name__ == "__main__":
    import json
    import os

    from georeset.fetchers.data_fetcher import DataFetcher

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = DataPaths()
    logger.info("Loading Wikipedia articles...")
    articles_path = paths.wiki_articles
    if os.path.exists(articles_path):
        with open(articles_path) as f:
            articles = json.load(f)
        logger.info("Loaded %s articles.", len(articles))
    else:
        articles = []
        logger.info("No articles found.")

    logger.info("Loading Alsace polygons...")
    fetcher = DataFetcher()
    gdf = fetcher.load_data(exclude_artificial=True)
    logger.info("Loaded %s polygons.", len(gdf))

    logger.info("Generating map...")
    viz = MapVisualizer(gdf)
    viz.save_map(paths.map_default, articles=articles)
    logger.info("Map saved to %s", paths.map_default)
