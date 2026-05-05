"""Tests for WikiFetcher."""

from unittest.mock import patch
from src.wiki_fetcher import WikiFetcher


class TestWikiFetcher:
    def setup_method(self):
        self.fetcher = WikiFetcher()

    def test_get_articles_in_bbox_returns_list(self):
        """Should return a list of articles within bounding box."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.headers = {"Content-Type": "application/json"}
            mock_get.return_value.json.return_value = {
                "query": {
                    "geosearch": [
                        {"pageid": 1, "title": "Strasbourg", "lat": 48.5, "lon": 7.5},
                        {"pageid": 2, "title": "Mulhouse", "lat": 48.1, "lon": 7.3},
                    ]
                }
            }
            articles = self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0)
            assert isinstance(articles, list)
            assert len(articles) == 2

    @patch("requests.get")
    def test_get_articles_in_bbox_handles_large_results_with_continuation(self, mock_get):
        """Should handle tiles with more than 500 articles via continuation."""
        # First call returns 500 articles + continuation
        # Second call returns remaining articles
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.side_effect = [
            {
                "query": {
                    "geosearch": [
                        {"pageid": i, "title": f"Article{i}", "lat": 48.5, "lon": 7.5}
                        for i in range(500)
                    ]
                },
                "continue": {"gscontinue": "next_page_token"}
            },
            {
                "query": {
                    "geosearch": [
                        {"pageid": 500 + i, "title": f"Article{500 + i}", "lat": 48.5, "lon": 7.5}
                        for i in range(100)
                    ]
                }
            }
        ]

        articles = self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0)

        # Should have made 2 calls (initial + continuation)
        assert mock_get.call_count == 2, f"Expected 2 calls for continuation, got {mock_get.call_count}"
        assert len(articles) == 600

    @patch("requests.get")
    def test_get_articles_in_bbox_filters_outside_bounds(self, mock_get):
        """get_articles_in_bbox should only return articles within the bbox (API already filters)."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "geosearch": [
                    {"pageid": 1, "title": "Inside", "lat": 48.5, "lon": 7.5},
                ]
            }
        }
        articles = self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0)
        titles = [a["title"] for a in articles]
        assert "Inside" in titles
        assert len(articles) == 1

    @patch("requests.get")
    def test_get_articles_in_bbox_uses_correct_params(self, mock_get):
        """Should call API with correct gsbbox parameter."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {"query": {"geosearch": []}}

        self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0)

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert "gsbbox" in params
        assert params["gsbbox"] == "49.0|7.0|48.0|8.0"

    @patch("requests.get")
    def test_get_articles_in_bbox_handles_empty_results(self, mock_get):
        """Should return empty list when no articles found."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {"query": {"geosearch": []}}
        articles = self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0)
        assert isinstance(articles, list)
        assert len(articles) == 0

    @patch("requests.get")
    def test_get_articles_in_bounds_uses_bbox_method(self, mock_get):
        """get_articles_in_bounds should use gsbbox internally via tiling."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
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
            "max_lon": 7.2,
            "max_lat": 48.2
        }
        self.fetcher.get_articles_in_bounds(**bounds)

        # Should call with gsbbox format
        assert mock_get.called
        # Verify at least one call used gsbbox
        calls_with_bbox = [c for c in mock_get.call_args_list if "gsbbox" in str(c)]
        assert len(calls_with_bbox) > 0

    def test_tiling_covers_full_bounds(self):
        """Verify the actual code's tiling covers the full bounds area."""
        min_lon, min_lat = 7.0, 48.0
        max_lon, max_lat = 8.0, 49.0

        # Capture tiles used by the actual code
        captured_tiles = []

        original_get_bbox = self.fetcher.get_articles_in_bbox

        def mock_get_bbox(north, west, south, east):
            captured_tiles.append((south, north, west, east))
            return []

        self.fetcher.get_articles_in_bbox = mock_get_bbox

        try:
            self.fetcher.get_articles_in_bounds(min_lon, min_lat, max_lon, max_lat)
        finally:
            self.fetcher.get_articles_in_bbox = original_get_bbox

        # Verify all points are covered by captured tiles
        uncovered = []
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                covered = False
                for t_south, t_north, t_west, t_east in captured_tiles:
                    if t_south <= lat <= t_north and t_west <= lon <= t_east:
                        covered = True
                        break
                if not covered:
                    uncovered.append((lat, lon))
                lon += 0.01
            lat += 0.01

        assert len(uncovered) == 0, f"Found {len(uncovered)} uncovered points. Tiles used: {len(captured_tiles)}"

    def test_get_nearby_articles_handles_no_results(self):
        """Should return empty list when no articles found."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.headers = {"Content-Type": "application/json"}
            mock_get.return_value.json.return_value = {"query": {"geosearch": []}}
            articles = self.fetcher.get_nearby_articles(0.0, 0.0)
            assert isinstance(articles, list)
            assert len(articles) == 0
