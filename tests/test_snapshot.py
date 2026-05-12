"""Tests for snapshot."""

from unittest.mock import MagicMock, patch

import pandas as pd

from georeset.cli.dev.snapshot import main, snapshot


def _mock_gdf():
    return pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "code_18": ["111", "112", "121", "131"],
        }
    )


def _mock_sample():
    return pd.DataFrame(
        {
            "class_label": ["11", "12"],
            "code_18": ["111", "121"],
            "centroid": [MagicMock(x=2.3, y=48.1), MagicMock(x=2.56, y=48.98)],
        }
    )


class TestSnapshot:
    @patch("georeset.cli.dev.snapshot.DataFetcher")
    def test_snapshot_runs_without_error(self, mock_fetcher_class):
        """Should run without raising exceptions when mocked."""
        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = _mock_gdf()
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = _mock_sample()

        snapshot(n_samples=3)

    @patch("georeset.cli.dev.snapshot.DataFetcher")
    def test_snapshot_calls_load_data(self, mock_fetcher_class):
        """Should load CORINE with artificial surfaces excluded."""
        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = _mock_gdf()
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = _mock_sample()

        snapshot(n_samples=1)
        mock_fetcher.load_data.assert_called_once_with(exclude_artificial=True)

    @patch("georeset.cli.dev.snapshot.DataFetcher")
    def test_snapshot_calls_get_bounds(self, mock_fetcher_class):
        """Should call get_bounds on the DataFetcher."""
        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = _mock_gdf()
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = _mock_sample()

        snapshot(n_samples=1)
        mock_fetcher.get_bounds.assert_called_once()

    @patch("georeset.cli.dev.snapshot.DataFetcher")
    def test_snapshot_calls_get_sample_polygons(self, mock_fetcher_class):
        """Should call get_sample_polygons with correct n_samples."""
        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = _mock_gdf()
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 8.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = _mock_sample()

        snapshot(n_samples=5)
        mock_fetcher.get_sample_polygons.assert_called_once_with(
            n=5, level=2, exclude_artificial=True
        )

    def test_snapshot_returns_formatted_report(self):
        """Should return the same snapshot sections as human-readable output."""
        with patch("georeset.cli.dev.snapshot.DataFetcher") as mock_fetcher_class:
            mock_fetcher = mock_fetcher_class.return_value
            mock_fetcher.load_data.return_value = _mock_gdf()
            mock_fetcher.get_bounds.return_value = (7.0, 48.0, 9.0, 49.0)
            mock_fetcher.get_sample_polygons.return_value = _mock_sample()

            report = snapshot(n_samples=2)

        assert isinstance(report, str)
        assert "=== Dataset Snapshot ===" in report
        assert "Total polygons: 4" in report
        assert "Bounds: (7.0, 48.0, 9.0, 49.0)" in report
        assert "Columns: ['ID', 'code_18']" in report
        assert "--- Level 1 Class Distribution ---" in report
        assert "--- Polygons by Number of Unique Classes ---" in report
        assert "--- Sample Polygons ---" in report
        assert "Class: 11, Code: 111" in report
        assert "Class: 12, Code: 121" in report

    @patch("georeset.cli.dev.snapshot.DataFetcher")
    @patch("builtins.print")
    def test_snapshot_does_not_print(self, mock_print, mock_fetcher_class):
        """Reusable snapshot helper should return text instead of printing."""
        mock_fetcher = mock_fetcher_class.return_value
        mock_fetcher.load_data.return_value = _mock_gdf()
        mock_fetcher.get_bounds.return_value = (7.0, 48.0, 9.0, 49.0)
        mock_fetcher.get_sample_polygons.return_value = _mock_sample()

        snapshot(n_samples=1)
        assert not mock_print.called

    @patch("builtins.print")
    @patch("georeset.cli.dev.snapshot.snapshot")
    def test_main_prints_snapshot_once(self, mock_snapshot, mock_print):
        """Main should print snapshot output exactly once."""
        mock_snapshot.return_value = "SNAPSHOT_TEXT"

        main(["--n-samples", "3"])

        mock_snapshot.assert_called_once_with(n_samples=3)
        mock_print.assert_called_once_with("SNAPSHOT_TEXT")
