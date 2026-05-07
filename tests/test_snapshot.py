"""Tests for snapshot."""

from unittest.mock import MagicMock, patch

from scripts.dev.snapshot import snapshot


class TestSnapshot:
    @patch("scripts.dev.snapshot.DataFetcher")
    def test_snapshot_runs_without_error(self, mock_fetcher_class):
        """Should run without raising exceptions when mocked."""
        mock_gdf = MagicMock()
        mock_gdf.__len__.return_value = 100
        mock_gdf.columns = ["ID", "code_18"]

        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = mock_gdf
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = mock_gdf

        with patch.object(mock_gdf, "__getitem__", side_effect=lambda key: MagicMock()):
            snapshot(n_samples=3)

    @patch("scripts.dev.snapshot.DataFetcher")
    def test_snapshot_calls_load_data(self, mock_fetcher_class):
        """Should call load_data on the DataFetcher."""
        mock_gdf = MagicMock()
        mock_gdf.__len__.return_value = 10
        mock_gdf.columns = ["ID", "code_18"]
        mock_gdf.__getitem__ = lambda self, key: MagicMock()

        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = mock_gdf
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = mock_gdf

        snapshot(n_samples=1)
        mock_fetcher.load_data.assert_called_once()

    @patch("scripts.dev.snapshot.DataFetcher")
    def test_snapshot_calls_get_bounds(self, mock_fetcher_class):
        """Should call get_bounds on the DataFetcher."""
        mock_gdf = MagicMock()
        mock_gdf.__len__.return_value = 10
        mock_gdf.columns = ["ID", "code_18"]
        mock_gdf.__getitem__ = lambda self, key: MagicMock()

        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = mock_gdf
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = mock_gdf

        snapshot(n_samples=1)
        mock_fetcher.get_bounds.assert_called_once()

    @patch("scripts.dev.snapshot.DataFetcher")
    def test_snapshot_calls_get_sample_polygons(self, mock_fetcher_class):
        """Should call get_sample_polygons with correct n_samples."""
        mock_gdf = MagicMock()
        mock_gdf.__len__.return_value = 10
        mock_gdf.columns = ["ID", "code_18"]
        mock_gdf.__getitem__ = lambda self, key: MagicMock()

        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = mock_gdf
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = mock_gdf

        snapshot(n_samples=5)
        mock_fetcher.get_sample_polygons.assert_called_once_with(n=5, level=2)

    @patch("builtins.print")
    @patch("scripts.dev.snapshot.DataFetcher")
    def test_snapshot_prints_dataset_info(self, mock_fetcher_class, mock_print):
        """Should print dataset information."""
        mock_gdf = MagicMock()
        mock_gdf.__len__.return_value = 42
        mock_gdf.columns = ["ID", "code_18"]
        mock_gdf.__getitem__ = lambda self, key: MagicMock()

        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = mock_gdf
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = mock_gdf

        snapshot(n_samples=1)
        # Verify print was called at least once (snapshot prints headers)
        assert mock_print.called
