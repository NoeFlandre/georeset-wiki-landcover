"""Tests for MapVisualizer."""

import geopandas as gpd
from shapely.geometry import Polygon

from georeset_wiki_landcover.visualization.map_visualizer import MapVisualizer


class TestMapVisualizer:
    def setup_method(self):
        self.basic_polygon = Polygon([(7.0, 48.0), (7.1, 48.0), (7.1, 48.1), (7.0, 48.1)])
        self.gdf = gpd.GeoDataFrame(
            {
                "class_label": ["311", "312"],
                "code_18": ["311", "312"],
                "geometry": [self.basic_polygon, self.basic_polygon],
            },
            crs="EPSG:4326",
        )
        self.articles = [
            {"title": "Strasbourg", "lat": 48.5, "lon": 7.5},
            {"title": "Mulhouse", "lat": 48.1, "lon": 7.3},
        ]

    def test_map_visualizer_initialization(self):
        """Should initialize with a GeoDataFrame."""
        visualizer = MapVisualizer(self.gdf)
        assert visualizer.gdf is not None

    def test_plot_polygons_returns_folium_map(self):
        """Should return a folium Map object."""
        import folium

        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons()
        assert isinstance(m, folium.Map)

    def test_plot_polygons_handles_empty_geodataframe(self):
        """Empty geospatial inputs should still produce a usable base map."""
        import folium

        empty = gpd.GeoDataFrame([], columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        visualizer = MapVisualizer(empty)

        m = visualizer.plot_polygons()

        assert isinstance(m, folium.Map)
        html = m._repr_html_()
        assert "leaflet" in html.lower()

    def test_save_map_creates_html_file(self, tmp_path):
        """Should save the map as an HTML file."""
        output_path = tmp_path / "test_map.html"
        visualizer = MapVisualizer(self.gdf)
        visualizer.save_map(str(output_path))
        assert output_path.exists()

    def test_save_map_creates_missing_parent_directories(self, tmp_path):
        """Map saving should own parent directory creation."""
        output_path = tmp_path / "nested" / "maps" / "test_map.html"
        visualizer = MapVisualizer(self.gdf)

        visualizer.save_map(str(output_path))

        assert output_path.exists()

    def test_plot_polygons_adds_polygon_layers(self):
        """Should add GeoJson polygon layers to the map."""
        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons()
        # Check that there are child objects (polygon layers)
        assert len(m._children) > 0

    def test_plot_polygons_with_articles_returns_map(self):
        """Should return a folium Map with article markers."""
        import folium

        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons_with_articles(self.articles)
        assert isinstance(m, folium.Map)

    def test_plot_polygons_with_articles_adds_markers(self):
        """Should add CircleMarker layers for articles."""
        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons_with_articles(self.articles)
        # Check that there are more children than polygons alone
        assert len(m._children) > len(self.gdf)

    def test_plot_polygons_with_articles_has_legend(self):
        """Should include legend showing polygon and article counts."""
        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons_with_articles(self.articles)
        html = m._repr_html_()
        assert "Polygons" in html
        assert "Articles" in html
        assert str(len(self.gdf)) in html
        assert str(len(self.articles)) in html

    def test_plot_corine_with_osm_polygons_has_both_layers(self):
        """Should render separate layers for CORINE and OSM polygons."""
        osm_gdf = gpd.GeoDataFrame(
            {
                "osm_id": ["way/1"],
                "geometry": [Polygon([(7.02, 48.02), (7.08, 48.02), (7.08, 48.08), (7.02, 48.08)])],
            },
            crs="EPSG:4326",
        )
        visualizer = MapVisualizer(self.gdf)

        m = visualizer.plot_corine_with_osm_polygons(osm_gdf)

        html = m._repr_html_()
        assert "CORINE polygons" in html
        assert "OSM polygons" in html

    def test_plot_corine_with_osm_polygons_handles_empty_corine_layer(self):
        import folium

        empty_corine = gpd.GeoDataFrame(
            [], columns=["code_18", "geometry"], geometry="geometry", crs="EPSG:4326"
        )
        osm_gdf = gpd.GeoDataFrame(
            {
                "osm_id": ["way/1"],
                "geometry": [Polygon([(7.02, 48.02), (7.08, 48.02), (7.08, 48.08), (7.02, 48.08)])],
            },
            crs="EPSG:4326",
        )
        visualizer = MapVisualizer(empty_corine)

        m = visualizer.plot_corine_with_osm_polygons(osm_gdf)

        assert isinstance(m, folium.Map)
        assert "OSM polygons" in m._repr_html_()

    def test_plot_corine_with_osm_polygons_legend_has_counts(self):
        osm_gdf = gpd.GeoDataFrame(
            {
                "osm_id": ["way/1"],
                "geometry": [Polygon([(7.02, 48.02), (7.08, 48.02), (7.08, 48.08), (7.02, 48.08)])],
            },
            crs="EPSG:4326",
        )
        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_corine_with_osm_polygons(osm_gdf)
        html = m._repr_html_()
        assert f"CORINE: {len(self.gdf)}" in html
        assert f"OSM: {len(osm_gdf)}" in html

    def test_plot_polygons_with_articles_legend_counts_match_data(self):
        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons_with_articles(self.articles)
        html = m._repr_html_()
        assert str(len(self.gdf)) in html
        assert str(len(self.articles)) in html
