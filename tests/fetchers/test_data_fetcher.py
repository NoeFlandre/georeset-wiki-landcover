"""Tests for DataFetcher."""

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import Point

from georeset_wiki_landcover.fetchers.data_fetcher import DataFetcher
from georeset_wiki_landcover.utils.json_io import read_json_file


def _write_corine_geojson(path: Path, codes: list[str], *, crs: str = "EPSG:4326") -> None:
    gdf = gpd.GeoDataFrame(
        {"code_18": codes},
        geometry=[Point(7.0 + index / 10, 48.0 + index / 10) for index, _ in enumerate(codes)],
        crs=crs,
    )
    gdf.to_file(path, driver="GeoJSON")


@pytest.fixture
def corine_fixture_path(tmp_path: Path) -> Path:
    """Create a compact CORINE-like dataset for hermetic DataFetcher tests."""
    data_path = tmp_path / "corine.geojson"
    _write_corine_geojson(data_path, ["111", "211", "221", "311", "312"])
    return data_path


class TestDataFetcher:
    def test_load_data_returns_geodataframe(self, corine_fixture_path: Path):
        """Should load and return a GeoDataFrame."""
        gdf = DataFetcher(data_path=corine_fixture_path).load_data()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_load_data_raises_on_missing_file(self):
        """Should raise FileNotFoundError when dataset is missing."""
        fetcher = DataFetcher(data_path="nonexistent/path.shp")
        with pytest.raises(FileNotFoundError):
            fetcher.load_data()

    def test_load_data_raises_clear_error_when_code_column_missing(self, tmp_path: Path):
        data_path = tmp_path / "missing_code.geojson"
        gdf = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(7.0, 48.0)], crs="EPSG:4326")
        gdf.to_file(data_path, driver="GeoJSON")

        fetcher = DataFetcher(data_path=str(data_path))

        with pytest.raises(ValueError, match="code_18"):
            fetcher.load_data()

    def test_load_data_reprojects_to_wgs84(self, tmp_path: Path):
        data_path = tmp_path / "lambert.geojson"
        gdf = gpd.GeoDataFrame(
            {"code_18": ["311"]},
            geometry=[Point(1_040_000, 6_840_000)],
            crs="EPSG:2154",
        )
        gdf.to_file(data_path, driver="GeoJSON")

        fetcher = DataFetcher(data_path=str(data_path))

        loaded = fetcher.load_data()

        assert loaded.crs == "EPSG:4326"

    def test_load_data_accepts_pathlike_data_path(self, tmp_path: Path):
        data_path = tmp_path / "corine.geojson"
        _write_corine_geojson(data_path, ["311"])
        fetcher = DataFetcher(data_path=data_path)

        loaded = fetcher.load_data()

        assert loaded["code_18"].tolist() == ["311"]

    def test_load_data_uses_logging_not_print(self, tmp_path: Path):
        data_path = tmp_path / "corine.geojson"
        _write_corine_geojson(data_path, ["311"])
        fetcher = DataFetcher(data_path=str(data_path))

        with patch("builtins.print") as print_mock:
            fetcher.load_data()

        print_mock.assert_not_called()

    def test_get_sample_polygons_raises_clear_error_when_request_exceeds_available(
        self, tmp_path: Path
    ):
        data_path = tmp_path / "small.geojson"
        _write_corine_geojson(data_path, ["311"])
        fetcher = DataFetcher(data_path=str(data_path))

        with pytest.raises(ValueError, match="Cannot sample 2 polygons from 1 available"):
            fetcher.get_sample_polygons(n=2)

    def test_get_sample_polygons_reports_filtering_when_no_non_artificial_polygons(
        self, tmp_path: Path
    ):
        data_path = tmp_path / "artificial.geojson"
        _write_corine_geojson(data_path, ["111"])
        fetcher = DataFetcher(data_path=str(data_path))

        with pytest.raises(ValueError, match="after filtering artificial surfaces"):
            fetcher.get_sample_polygons(n=1, exclude_artificial=True)

    def test_get_sample_polygons_fails_clearly_when_loader_does_not_populate_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fetcher = DataFetcher(data_path="unused.geojson")
        monkeypatch.setattr(
            fetcher, "load_data", lambda exclude_artificial=False: gpd.GeoDataFrame()
        )

        with pytest.raises(RuntimeError, match="Dataset could not be loaded"):
            fetcher.get_sample_polygons(n=1)

    def test_get_sample_polygons_returns_geodataframe(self, corine_fixture_path: Path):
        """Should return a GeoDataFrame with sampled polygons."""
        sample = DataFetcher(data_path=corine_fixture_path).get_sample_polygons(n=3)
        assert isinstance(sample, gpd.GeoDataFrame)
        assert len(sample) == 3

    def test_get_sample_polygons_includes_class_label(self, corine_fixture_path: Path):
        """Sample should include class_label column."""
        sample = DataFetcher(data_path=corine_fixture_path).get_sample_polygons(n=3)
        assert "class_label" in sample.columns

    def test_get_sample_polygons_includes_centroid(self, corine_fixture_path: Path):
        """Sample should include centroid column."""
        sample = DataFetcher(data_path=corine_fixture_path).get_sample_polygons(n=3)
        assert "centroid" in sample.columns

    def test_get_sample_polygons_level_affects_class_label_length(self, corine_fixture_path: Path):
        """Class label length should match the specified level."""
        fetcher = DataFetcher(data_path=corine_fixture_path)
        sample_level1 = fetcher.get_sample_polygons(n=5, level=1)
        sample_level2 = fetcher.get_sample_polygons(n=5, level=2)
        assert all(sample_level1["class_label"].str.len() == 1)
        assert all(sample_level2["class_label"].str.len() == 2)

    def test_get_bounds_returns_tuple(self, corine_fixture_path: Path):
        """Should return a tuple of 4 bounds."""
        bounds = DataFetcher(data_path=corine_fixture_path).get_bounds()
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4

    def test_get_bounds_values_are_valid_coordinates(self, corine_fixture_path: Path):
        """Bounds should be valid longitude/latitude values."""
        min_lon, min_lat, max_lon, max_lat = DataFetcher(data_path=corine_fixture_path).get_bounds()
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90
        assert -180 <= max_lon <= 180
        assert -90 <= max_lat <= 90
        assert min_lon <= max_lon
        assert min_lat <= max_lat

    def test_save_bounds_creates_file(self, tmp_path: Path, corine_fixture_path: Path):
        """Should save bounds to a JSON file."""
        output_path = tmp_path / "bounds.json"
        DataFetcher(data_path=corine_fixture_path).save_bounds(output_path)
        assert output_path.exists()

    def test_save_bounds_contains_correct_keys(self, tmp_path: Path, corine_fixture_path: Path):
        """Saved bounds JSON should have expected keys."""
        output_path = tmp_path / "bounds.json"
        DataFetcher(data_path=corine_fixture_path).save_bounds(output_path)

        data = read_json_file(output_path)
        assert "min_lon" in data
        assert "min_lat" in data
        assert "max_lon" in data
        assert "max_lat" in data

    def test_exclude_artificial_filters_class_1(self, corine_fixture_path: Path):
        """Should exclude polygons where code_18 starts with '1' (artificial surfaces)."""
        sample = DataFetcher(data_path=corine_fixture_path).get_sample_polygons(
            n=4, exclude_artificial=True
        )
        artificial_codes = sample[sample["code_18"].str.startswith("1")]
        assert len(artificial_codes) == 0, (
            f"Found artificial surface codes: {artificial_codes['code_18'].tolist()}"
        )

    def test_exclude_artificial_false_includes_class_1(self, corine_fixture_path: Path):
        """When exclude_artificial=False, polygons with code starting in '1' should be present."""
        sample = DataFetcher(data_path=corine_fixture_path).get_sample_polygons(
            n=5, exclude_artificial=False
        )
        artificial_codes = sample[sample["code_18"].str.startswith("1")]
        assert len(artificial_codes) > 0, "Expected some artificial surface codes in full sample"

    def test_load_data_exclude_artificial(self, corine_fixture_path: Path):
        """load_data should accept exclude_artificial flag."""
        gdf = DataFetcher(data_path=corine_fixture_path).load_data(exclude_artificial=True)
        artificial = gdf[gdf["code_18"].str.startswith("1")]
        assert len(artificial) == 0
