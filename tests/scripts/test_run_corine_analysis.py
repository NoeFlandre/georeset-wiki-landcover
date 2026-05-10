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
        return gpd.GeoDataFrame(
            {
                "code_18": ["111", "211", "311", "112"],
                "geometry": [
                    Polygon([(7.0, 48.0), (7.1, 48.0), (7.1, 48.1), (7.0, 48.1)]),
                    Polygon([(7.1, 48.0), (7.2, 48.0), (7.2, 48.1), (7.1, 48.1)]),
                    Polygon([(7.2, 48.0), (7.3, 48.0), (7.3, 48.1), (7.2, 48.1)]),
                    Polygon([(7.3, 48.0), (7.4, 48.0), (7.4, 48.1), (7.3, 48.1)]),
                ],
            },
            crs="EPSG:4326",
        )

    @pytest.fixture
    def mock_osm_gdf(self):
        """Create a mock OSM GeoDataFrame."""
        return gpd.GeoDataFrame(
            {
                "osm_id": ["way/1", "way/2", "way/3"],
                "landuse": ["forest", "meadow", "farmland"],
                "natural": [None, None, None],
                "geometry": [
                    Polygon([(7.05, 48.05), (7.15, 48.05), (7.15, 48.15), (7.05, 48.15)]),
                    Polygon([(7.15, 48.05), (7.25, 48.05), (7.25, 48.15), (7.15, 48.15)]),
                    Polygon([(7.25, 48.05), (7.35, 48.05), (7.35, 48.15), (7.25, 48.15)]),
                ],
            },
            crs="EPSG:4326",
        )

    def test_run_with_exclude_artificial_loads_filtered_corine(
        self, mock_corine_gdf, mock_osm_gdf, tmp_path
    ):
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
        osm_path.parent.mkdir()
        map_path.parent.mkdir()
        dist_path.parent.mkdir()

        osm_gdf = mock_osm_gdf.copy()
        corine_gdf = mock_corine_gdf.copy()

        with (
            patch("georeset.fetchers.data_fetcher.DataFetcher.load_data") as mock_load,
            patch("geopandas.read_file") as mock_read,
            patch("geopandas.GeoDataFrame.to_file") as _mock_to_file,
            patch("scripts.analysis.run_corine_analysis.write_csv_atomic") as _mock_write_csv,
            patch("folium.Map.save") as _mock_map_save,
        ):
            mock_load.return_value = corine_gdf
            mock_read.return_value = osm_gdf

            run(
                output_map_path=str(map_path),
                output_csv_path=str(dist_path),
                osm_polygons_path=str(osm_path),
                refetch_osm=False,
            )

            # Verify exclude_artificial=True was passed
            call_kwargs = mock_load.call_args
            assert "exclude_artificial" in call_kwargs.kwargs
            assert call_kwargs.kwargs["exclude_artificial"] is True

    def test_run_keeps_osm_polygons_overlapping_filtered_corine_only(self, tmp_path):
        """run() should keep OSM polygons that intersect filtered (non-artificial) CORINE,
        even if they also overlap artificial raw CORINE elsewhere in the raw layer."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "corine").mkdir()
        (data_dir / "corine" / "bounds.json").write_text(
            json.dumps({"min_lon": 0.0, "min_lat": 0.0, "max_lon": 30.0, "max_lat": 10.0})
        )
        osm_path = data_dir / "osm" / "osm_project_polygons.geojson"
        map_path = data_dir / "maps" / "osm_corine_polygons.html"
        dist_path = data_dir / "distribution" / "osm_corine_distribution.csv"
        osm_path.parent.mkdir()
        map_path.parent.mkdir()
        dist_path.parent.mkdir()

        full_corine = gpd.GeoDataFrame(
            {
                "code_18": ["112", "311", "211"],
                "geometry": [
                    Polygon([(0, 0), (2, 0), (2, 10), (0, 10)]),
                    Polygon([(2, 0), (10, 0), (10, 10), (2, 10)]),
                    Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
                ],
            },
            crs="EPSG:3857",
        )
        filtered_corine = full_corine[~full_corine["code_18"].str.startswith("1")].copy()
        osm = gpd.GeoDataFrame(
            {
                "osm_id": ["mixed", "natural"],
                "landuse": ["forest", "meadow"],
                "natural": [None, None],
                "geometry": [
                    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                    Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
                ],
            },
            crs="EPSG:3857",
        )
        written = {}

        def capture_to_file(gdf, path, *args, **kwargs):
            written["osm"] = gdf.copy()

        with (
            patch("georeset.fetchers.data_fetcher.DataFetcher.load_data") as mock_load,
            patch("geopandas.read_file", return_value=osm),
            patch("geopandas.GeoDataFrame.to_file", new=capture_to_file),
            patch("folium.Map.save"),
        ):
            mock_load.return_value = filtered_corine

            run(
                output_map_path=str(map_path),
                output_csv_path=str(dist_path),
                osm_polygons_path=str(osm_path),
                refetch_osm=False,
                chunk_size=1,
            )

        # 'mixed' overlaps artificial 112 AND natural 311 in raw; but filtered CORINE
        # only has 311, so 'mixed' should be kept as it overlaps filtered CORINE.
        assert written["osm"]["osm_id"].tolist() == ["mixed", "natural"]

    def test_run_fetches_osm_when_refetch_requested(self, mock_corine_gdf, mock_osm_gdf, tmp_path):
        bounds_path = tmp_path / "bounds.json"
        bounds_path.write_text(
            json.dumps({"min_lon": 7.0, "min_lat": 48.0, "max_lon": 8.0, "max_lat": 49.0})
        )
        osm_path = tmp_path / "osm" / "osm_project_polygons.geojson"
        map_path = tmp_path / "maps" / "osm_corine_polygons.html"
        dist_path = tmp_path / "distribution" / "osm_corine_distribution.csv"
        fetched = mock_osm_gdf.copy()

        with (
            patch(
                "georeset.fetchers.data_fetcher.DataFetcher.load_data", return_value=mock_corine_gdf
            ),
            patch("scripts.analysis.run_corine_analysis.OSMFetcher") as mock_fetcher_cls,
            patch("geopandas.read_file") as mock_read_file,
            patch("geopandas.GeoDataFrame.to_file"),
            patch("scripts.analysis.run_corine_analysis.write_csv_atomic"),
            patch("folium.Map.save"),
        ):
            mock_fetcher_cls.return_value.fetch_polygons.return_value = fetched

            run(
                output_map_path=str(map_path),
                output_csv_path=str(dist_path),
                osm_polygons_path=str(osm_path),
                corine_bounds_path=str(bounds_path),
                refetch_osm=True,
            )

        mock_fetcher_cls.return_value.fetch_polygons.assert_called_once_with(7.0, 48.0, 8.0, 49.0)
        mock_read_file.assert_not_called()

    def test_run_loads_existing_osm_when_refetch_not_requested(
        self, mock_corine_gdf, mock_osm_gdf, tmp_path
    ):
        bounds_path = tmp_path / "bounds.json"
        bounds_path.write_text(
            json.dumps({"min_lon": 7.0, "min_lat": 48.0, "max_lon": 8.0, "max_lat": 49.0})
        )
        osm_path = tmp_path / "osm" / "osm_project_polygons.geojson"
        map_path = tmp_path / "maps" / "osm_corine_polygons.html"
        dist_path = tmp_path / "distribution" / "osm_corine_distribution.csv"

        with (
            patch(
                "georeset.fetchers.data_fetcher.DataFetcher.load_data", return_value=mock_corine_gdf
            ),
            patch("geopandas.read_file", return_value=mock_osm_gdf) as mock_read_file,
            patch("scripts.analysis.run_corine_analysis.OSMFetcher") as mock_fetcher_cls,
            patch("geopandas.GeoDataFrame.to_file"),
            patch("scripts.analysis.run_corine_analysis.write_csv_atomic"),
            patch("folium.Map.save"),
        ):
            run(
                output_map_path=str(map_path),
                output_csv_path=str(dist_path),
                osm_polygons_path=str(osm_path),
                corine_bounds_path=str(bounds_path),
                refetch_osm=False,
            )

        mock_read_file.assert_called_once_with(str(osm_path))
        mock_fetcher_cls.return_value.fetch_polygons.assert_not_called()
