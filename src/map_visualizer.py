"""OSM map visualizer for Corine Land Cover polygons."""

import folium
import geopandas as gpd


class MapVisualizer:
    """Visualizes GeoDataFrame polygons on an OpenStreetMap."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf

    def plot_polygons(self, zoom_start: int = 8) -> folium.Map:
        """Plot polygons on an OSM map. Returns a folium Map."""
        bounds = self.gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

        for _, row in self.gdf.iterrows():
            folium.GeoJson(row["geometry"]).add_to(m)

        return m

    def plot_polygons_with_articles(self, articles: list, zoom_start: int = 8) -> folium.Map:
        """Plot polygons and Wikipedia article locations as dots."""
        bounds = self.gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

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
                popup=article.get("title", "Article")
            ).add_to(m)

        return m

    def save_map(self, output_path: str) -> None:
        """Save the map as an HTML file."""
        m = self.plot_polygons()
        m.save(output_path)
