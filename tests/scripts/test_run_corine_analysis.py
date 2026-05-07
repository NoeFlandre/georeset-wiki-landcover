"""Smoke test for run_corine_analysis script."""

import json
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from scripts.analysis.run_corine_analysis import run


def test_run_corine_analysis_exports_run():
    """Should have a run function."""
    assert callable(run)


class TestRunCorineAnalysisWithFilteredCorine:
    """Tests for run_corine_analysis with exclude_artificial=True."""

    @pytest.fixture
    def mock_corine_gdf(self):
        """Create a mock CORINE GeoDataFrame with mixed classes."""
        return gpd.GeoDataFrame({
            "code_18": ["111", "211", "311", "112"],
            "geometry": [
                Polygon([(7.0, 48.0), (7.1, 48.0), (7.1, 48.1), (7.0, 48.1)]),
                Polygon([(7.1, 48.0), (7.2, 48.0), (7.2, 48.1), (7.1, 48.1)]),
                Polygon([(7.2, 48.0), (7.3, 48.0), (7.3, 48.1), (7.2, 48.1)]),
                Polygon([(7.3, 48.0), (7.4, 48.0), (7.4, 48.1), (7.3, 48.1)]),
            ]
        }, crs="EPSG:4326")

    @pytest.fixture
    def mock_osm_gdf(self):
        """Create a mock OSM GeoDataFrame."""
        return gpd.GeoDataFrame({
            "osm_id": ["way/1", "way/2", "way/3"],
            "landuse": ["forest", "meadow", "farmland"],
            "natural": [None, None, None],
            "geometry": [
                Polygon([(7.05, 48.05), (7.15, 48.05), (7.15, 48.15), (7.05, 48.15)]),
                Polygon([(7.15, 48.05), (7.25, 48.05), (7.25, 48.15), (7.15, 48.15)]),
                Polygon([(7.25, 48.05), (7.35, 48.05), (7.35, 48.15), (7.25, 48.15)]),
            ]
        }, crs="EPSG:4326")

    def test_run_with_exclude_artificial_loads_filtered_corine(self, mock_corine_gdf, mock_osm_gdf, tmp_path):
        """run() should use exclude_artificial=True when loading CORINE."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "corine").mkdir()
        (data_dir / "corine" / "bounds.json").write_text(
            json.dumps({"min_lon": 7.0, "min_lat": 48.0, "max_lon": 8.0, "max_lat": 49.0})
        )
        osm_path = data_dir / "osm" / "osm_project_polygons.geojson"
        map_path = data_dir / "maps" / "osm_corine_polygons.html"
        dist_path = data_dir / "distribution" / "osm_corine_distribution.csv"

        osm_gdf = mock_osm_gdf.copy()
        corine_gdf = mock_corine_gdf.copy()

        with (
            patch("src.fetchers.data_fetcher.DataFetcher.load_data") as mock_load,
            patch("geopandas.read_file") as mock_read,
            patch("geopandas.GeoDataFrame.to_file") as _mock_to_file,
            patch("pandas.DataFrame.to_csv") as _mock_to_csv,
            patch("folium.Map.save") as _mock_map_save,
        ):
            mock_load.return_value = corine_gdf
            mock_read.return_value = osm_gdf

            run(
                output_map_path=str(map_path),
                output_csv_path=str(dist_path),
                output_osm_path=str(osm_path),
            )

            # Verify exclude_artificial=True was passed
            call_kwargs = mock_load.call_args
            assert "exclude_artificial" in call_kwargs.kwargs
            assert call_kwargs.kwargs["exclude_artificial"] is True
