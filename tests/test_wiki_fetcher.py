"""Tests for WikiFetcher."""

import pytest
from unittest.mock import patch, MagicMock
from src.wiki_fetcher import WikiFetcher


class TestWikiFetcher:
    def setup_method(self):
        self.fetcher = WikiFetcher()

    def test_get_nearby_articles_returns_list(self):
        """Should return a list of articles near a coordinate."""
        articles = self.fetcher.get_nearby_articles(48.8, 7.7)
        assert isinstance(articles, list)

    def test_get_nearby_articles_contains_article_structure(self):
        """Articles should have pageid, title, and coordinates."""
        articles = self.fetcher.get_nearby_articles(48.8, 7.7)
        if articles:
            article = articles[0]
            assert "pageid" in article
            assert "title" in article
            assert "lat" in article
            assert "lon" in article

    @patch("requests.get")
    def test_get_articles_in_bounds_returns_list(self, mock_get):
        """Should return a list of articles within bounding box."""
        mock_get.return_value.json.return_value = {
            "query": {
                "geosearch": [
                    {"pageid": 1, "title": "Strasbourg", "lat": 48.5, "lon": 7.5},
                    {"pageid": 2, "title": "Mulhouse", "lat": 48.1, "lon": 7.3},
                ]
            }
        }
        bounds = {
            "min_lon": 7.0,
            "min_lat": 48.0,
            "max_lon": 8.0,
            "max_lat": 49.0
        }
        articles = self.fetcher.get_articles_in_bounds(**bounds)
        assert isinstance(articles, list)
        assert len(articles) == 2

    @patch("requests.get")
    def test_get_articles_in_bounds_deduplicates(self, mock_get):
        """Should not return duplicate articles when circles overlap."""
        mock_get.return_value.json.return_value = {
            "query": {
                "geosearch": [
                    {"pageid": 1, "title": "Strasbourg", "lat": 48.5, "lon": 7.5},
                ]
            }
        }
        bounds = {
            "min_lon": 7.0,
            "min_lat": 48.0,
            "max_lon": 7.5,
            "max_lat": 48.5
        }
        articles = self.fetcher.get_articles_in_bounds(**bounds)
        pageids = [a["pageid"] for a in articles]
        assert len(pageids) == len(set(pageids)), "Duplicate pageids found"

    @patch("requests.get")
    def test_get_articles_in_bounds_filters_outside_bounds(self, mock_get):
        """Should not return articles outside the bounding box."""
        mock_get.return_value.json.return_value = {
            "query": {
                "geosearch": [
                    {"pageid": 1, "title": "Inside", "lat": 48.5, "lon": 7.5},  # inside bounds
                    {"pageid": 2, "title": "Outside", "lat": 46.0, "lon": 9.0},  # outside bounds
                ]
            }
        }
        bounds = {
            "min_lon": 7.0,
            "min_lat": 48.0,
            "max_lon": 8.0,
            "max_lat": 49.0
        }
        articles = self.fetcher.get_articles_in_bounds(**bounds)
        titles = [a["title"] for a in articles]
        assert "Inside" in titles
        assert "Outside" not in titles

    @patch("requests.get")
    def test_get_articles_in_bounds_grid_coverage(self, mock_get):
        """Should make multiple API calls to cover the full bounds area."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"query": {"geosearch": []}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        bounds = {
            "min_lon": 7.0,
            "min_lat": 48.0,
            "max_lon": 7.5,
            "max_lat": 48.5
        }
        self.fetcher.get_articles_in_bounds(**bounds, radius=10000)

        # With 10km radius and 80% overlap, this ~0.5° x 0.5° area should need multiple calls
        assert mock_get.call_count >= 4, f"Expected at least 4 API calls for coverage, got {mock_get.call_count}"

    def test_grid_coverage_is_complete(self):
        """Verify that the grid used by get_articles_in_bounds covers the full bounds."""
        import math
        from unittest.mock import patch, MagicMock

        min_lon, min_lat = 7.0, 48.0
        max_lon, max_lat = 8.0, 49.0
        radius = 10000

        # Capture the grid points that get_articles_in_bounds actually uses
        captured_points = []

        original_get_nearby = self.fetcher.get_nearby_articles

        def mock_get_nearby(lat, lon, r):
            captured_points.append((lat, lon))
            return []

        self.fetcher.get_nearby_articles = mock_get_nearby

        try:
            self.fetcher.get_articles_in_bounds(min_lon, min_lat, max_lon, max_lat, radius)
        finally:
            self.fetcher.get_nearby_articles = original_get_nearby

        # Verify every point in bounds is within radius of at least one captured grid point
        uncovered = []
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                covered = False
                for g_lat, g_lon in captured_points:
                    R = 6371000
                    dlat = math.radians(lat - g_lat)
                    dlon = math.radians(lon - g_lon)
                    a = math.sin(dlat/2)**2 + math.cos(math.radians(g_lat)) * math.cos(math.radians(lat)) * math.sin(dlon/2)**2
                    dist = 2 * R * math.asin(math.sqrt(a))
                    if dist <= radius:
                        covered = True
                        break
                if not covered:
                    uncovered.append((lat, lon))
                lon += 0.01
            lat += 0.01

        assert len(uncovered) == 0, f"Found {len(uncovered)} uncovered points. Grid points used: {len(captured_points)}"

    def test_get_nearby_articles_handles_no_results(self):
        """Should return empty list when no articles found."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"query": {"geosearch": []}}
            articles = self.fetcher.get_nearby_articles(0.0, 0.0)
            assert isinstance(articles, list)
            assert len(articles) == 0
