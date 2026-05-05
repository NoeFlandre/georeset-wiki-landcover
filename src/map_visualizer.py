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

        legend_html = f"""
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                    background-color: white; padding: 10px; border: 2px solid gray;
                    font-size: 14px; border-radius: 5px;">
            <b>Legend</b><br>
            <i style="background-color: blue; width: 20px; height: 10px; display: inline-block;"></i> Polygons: {len(self.gdf)}<br>
            <i style="background-color: red; width: 10px; height: 10px; display: inline-block; border-radius: 50%;"></i> Articles: {len(articles)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def save_map(self, output_path: str, articles: list = None) -> None:
        """Save the map as an HTML file."""
        if articles:
            m = self.plot_polygons_with_articles(articles)
        else:
            m = self.plot_polygons()
        m.save(output_path)


if __name__ == "__main__":
    import json
    import os
    from src.data_fetcher import DataFetcher

    print("Loading Wikipedia articles...")
    articles_path = "data/wiki_articles.json"
    if os.path.exists(articles_path):
        with open(articles_path, "r") as f:
            articles = json.load(f)
        print(f"Loaded {len(articles)} articles.")
    else:
        articles = []
        print("No articles found.")

    print("Loading Alsace polygons...")
    fetcher = DataFetcher()
    gdf = fetcher.load_data()
    print(f"Loaded {len(gdf)} polygons.")

    print("Generating map...")
    viz = MapVisualizer(gdf)
    viz.save_map("data/map.html", articles=articles)
    print("Map saved to data/map.html")
