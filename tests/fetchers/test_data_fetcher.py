"""Tests for DataFetcher."""

import geopandas as gpd
import pytest

from src.fetchers.data_fetcher import DataFetcher


class TestDataFetcher:
    def setup_method(self):
        self.fetcher = DataFetcher()

    def test_load_data_returns_geodataframe(self):
        """Should load and return a GeoDataFrame."""
        gdf = self.fetcher.load_data()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_load_data_raises_on_missing_file(self):
        """Should raise FileNotFoundError when dataset is missing."""
        fetcher = DataFetcher(data_path="nonexistent/path.shp")
        with pytest.raises(FileNotFoundError):
            fetcher.load_data()

    def test_get_sample_polygons_returns_geodataframe(self):
        """Should return a GeoDataFrame with sampled polygons."""
        sample = self.fetcher.get_sample_polygons(n=3)
        assert isinstance(sample, gpd.GeoDataFrame)
        assert len(sample) == 3

    def test_get_sample_polygons_includes_class_label(self):
        """Sample should include class_label column."""
        sample = self.fetcher.get_sample_polygons(n=3)
        assert "class_label" in sample.columns

    def test_get_sample_polygons_includes_centroid(self):
        """Sample should include centroid column."""
        sample = self.fetcher.get_sample_polygons(n=3)
        assert "centroid" in sample.columns

    def test_get_sample_polygons_level_affects_class_label_length(self):
        """Class label length should match the specified level."""
        sample_level1 = self.fetcher.get_sample_polygons(n=5, level=1)
        sample_level2 = self.fetcher.get_sample_polygons(n=5, level=2)
        assert all(sample_level1["class_label"].str.len() == 1)
        assert all(sample_level2["class_label"].str.len() == 2)

    def test_get_bounds_returns_tuple(self):
        """Should return a tuple of 4 bounds."""
        bounds = self.fetcher.get_bounds()
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4

    def test_get_bounds_values_are_valid_coordinates(self):
        """Bounds should be valid longitude/latitude values."""
        min_lon, min_lat, max_lon, max_lat = self.fetcher.get_bounds()
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90
        assert -180 <= max_lon <= 180
        assert -90 <= max_lat <= 90
        assert min_lon <= max_lon
        assert min_lat <= max_lat

    def test_save_bounds_creates_file(self, tmp_path):
        """Should save bounds to a JSON file."""
        output_path = tmp_path / "bounds.json"
        self.fetcher.save_bounds(str(output_path))
        assert output_path.exists()

    def test_save_bounds_contains_correct_keys(self, tmp_path):
        """Saved bounds JSON should have expected keys."""
        output_path = tmp_path / "bounds.json"
        self.fetcher.save_bounds(str(output_path))
        import json

        with open(output_path) as f:
            data = json.load(f)
        assert "min_lon" in data
        assert "min_lat" in data
        assert "max_lon" in data
        assert "max_lat" in data
