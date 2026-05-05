"""Tests for MapVisualizer."""

import pytest
import geopandas as gpd
from shapely.geometry import Polygon
from src.map_visualizer import MapVisualizer


class TestMapVisualizer:
    def setup_method(self):
        self.basic_polygon = Polygon([(7.0, 48.0), (7.1, 48.0), (7.1, 48.1), (7.0, 48.1)])
        self.gdf = gpd.GeoDataFrame({
            "class_label": ["311", "312"],
            "code_18": ["311", "312"],
            "geometry": [self.basic_polygon, self.basic_polygon]
        }, crs="EPSG:4326")

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

    def test_save_map_creates_html_file(self, tmp_path):
        """Should save the map as an HTML file."""
        output_path = tmp_path / "test_map.html"
        visualizer = MapVisualizer(self.gdf)
        visualizer.save_map(str(output_path))
        assert output_path.exists()

    def test_plot_polygons_adds_polygon_layers(self):
        """Should add GeoJson polygon layers to the map."""
        visualizer = MapVisualizer(self.gdf)
        m = visualizer.plot_polygons()
        # Check that there are child objects (polygon layers)
        assert len(m._children) > 0
