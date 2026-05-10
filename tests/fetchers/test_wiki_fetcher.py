"""Tests for WikiFetcher."""

from unittest.mock import patch

import pytest
import requests

from src.fetchers.wiki_fetcher import WikiFetcher, WikiFetchError


class TestWikiFetcher:
    def setup_method(self):
        self.fetcher = WikiFetcher()
        self.sleep_patcher = patch("src.fetchers.wiki_fetcher.time.sleep")
        self.sleep_patcher.start()

    def teardown_method(self):
        self.sleep_patcher.stop()

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
                "continue": {"gscontinue": "next_page_token"},
            },
            {
                "query": {
                    "geosearch": [
                        {"pageid": 500 + i, "title": f"Article{500 + i}", "lat": 48.5, "lon": 7.5}
                        for i in range(100)
                    ]
                }
            },
        ]

        articles = self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0)

        # Should have made 2 calls (initial + continuation)
        assert mock_get.call_count == 2, (
            f"Expected 2 calls for continuation, got {mock_get.call_count}"
        )
        assert len(articles) == 600

    @patch("requests.get")
    def test_get_articles_in_bbox_raises_when_all_retries_fail(self, mock_get):
        """Should not silently return partial data when Wikipedia keeps failing."""
        mock_get.side_effect = RuntimeError("network down")

        with pytest.raises(WikiFetchError):
            self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0, retries=2)

        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_nearby_articles_raises_when_all_retries_fail(self, mock_get):
        """Nearby article fetches should use the same fail-fast policy as bbox fetches."""
        mock_get.side_effect = requests.RequestException("network down")

        with pytest.raises(WikiFetchError):
            self.fetcher.get_nearby_articles(lat=48.5, lon=7.5, retries=2)

        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_get_articles_in_bbox_raises_when_continuation_page_fails(self, mock_get):
        """Should fail the bbox rather than returning only the first page."""
        first_page = {
            "headers": {"Content-Type": "application/json"},
            "json": lambda: {
                "query": {
                    "geosearch": [
                        {"pageid": 1, "title": "First", "lat": 48.5, "lon": 7.5},
                    ]
                },
                "continue": {"gscontinue": "next_page_token"},
            },
        }

        class Response:
            def __init__(self, status_code=200, headers=None, json_func=None):
                self.status_code = status_code
                self.headers = headers or {}
                self._json_func = json_func or (lambda: {})
                self.text = ""

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError("http error")

            def json(self):
                return self._json_func()

        mock_get.side_effect = [
            Response(headers=first_page["headers"], json_func=first_page["json"]),
            RuntimeError("network down"),
            RuntimeError("still down"),
        ]

        with pytest.raises(WikiFetchError):
            self.fetcher.get_articles_in_bbox(north=49.0, west=7.0, south=48.0, east=8.0, retries=2)

    @patch("requests.get")
    def test_get_articles_in_bounds_filters_bounds_polygon_and_bad_coordinates(self, mock_get):
        """Should only keep unique articles with valid coordinates inside bounds and polygon."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "geosearch": [
                    {"pageid": 1, "title": "Inside", "lat": 48.1, "lon": 7.1},
                    {"pageid": 1, "title": "Duplicate", "lat": 48.1, "lon": 7.1},
                    {"pageid": 2, "title": "Outside bounds", "lat": 48.1, "lon": 9.0},
                    {"pageid": 3, "title": "Outside polygon", "lat": 48.15, "lon": 7.15},
                    {"pageid": 4, "title": "Missing longitude", "lat": 48.1},
                ]
            }
        }

        articles = self.fetcher.get_articles_in_bounds(
            min_lon=7.0,
            min_lat=48.0,
            max_lon=7.2,
            max_lat=48.2,
            polygon_filter=lambda lon, lat: lon < 7.12,
        )

        assert articles == [{"pageid": 1, "title": "Inside", "lat": 48.1, "lon": 7.1}]

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

    def test_sleep_before_retry_uses_logging_not_print(self):
        with patch("builtins.print") as print_mock:
            self.fetcher._sleep_before_retry("Rate limited", attempt=0, scale=1)

        print_mock.assert_not_called()

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
        bounds = {"min_lon": 7.0, "min_lat": 48.0, "max_lon": 7.2, "max_lat": 48.2}
        self.fetcher.get_articles_in_bounds(**bounds)

        # Should call with gsbbox format
        assert mock_get.called
        # Verify at least one call used gsbbox
        calls_with_bbox = [c for c in mock_get.call_args_list if "gsbbox" in str(c)]
        assert len(calls_with_bbox) > 0

    @patch("requests.get")
    def test_get_articles_in_bounds_uses_logging_not_print(self, mock_get):
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {"query": {"geosearch": []}}

        with patch("builtins.print") as print_mock:
            self.fetcher.get_articles_in_bounds(
                min_lon=7.0,
                min_lat=48.0,
                max_lon=7.05,
                max_lat=48.05,
            )

        print_mock.assert_not_called()

    @patch("requests.get")
    def test_get_articles_in_bounds_with_osm_filter(self, mock_get):
        """Should keep article if in OSM polygon (even if not in corine)."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "geosearch": [
                    {"pageid": 1, "title": "InCorineOnly", "lat": 48.1, "lon": 7.1},
                    {"pageid": 2, "title": "InOSMOnly", "lat": 48.2, "lon": 7.4},
                    {"pageid": 3, "title": "InBoth", "lat": 48.3, "lon": 7.15},
                    {
                        "pageid": 4,
                        "title": "InNeither",
                        "lat": 48.15,
                        "lon": 7.25,
                    },  # in the gap: not in corine range [7.0,7.2], not in osm range [7.4,7.5]
                ]
            }
        }

        # Corine filter: only articles with lon in [7.0, 7.2]
        def corine_filter(lon: float, lat: float) -> bool:
            return 7.0 <= lon <= 7.2

        # OSM filter: only articles with lon in [7.4, 7.5]
        def osm_filter(lon: float, lat: float) -> bool:
            return 7.4 <= lon <= 7.5

        articles = self.fetcher.get_articles_in_bounds(
            min_lon=7.0,
            min_lat=48.0,
            max_lon=7.5,
            max_lat=48.5,
            polygon_filter=corine_filter,
            osm_polygon_filter=osm_filter,
        )

        pageids = [a["pageid"] for a in articles]
        # InCorineOnly (7.1): in corine -> kept
        # InOSMOnly (7.4): in osm -> kept
        # InBoth (7.15): in both -> kept
        # InNeither (7.25): in neither -> dropped
        assert 1 in pageids
        assert 2 in pageids
        assert 3 in pageids
        assert 4 not in pageids

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

        assert len(uncovered) == 0, (
            f"Found {len(uncovered)} uncovered points. Tiles used: {len(captured_tiles)}"
        )

    def test_get_nearby_articles_handles_no_results(self):
        """Should return empty list when no articles found."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.headers = {"Content-Type": "application/json"}
            mock_get.return_value.json.return_value = {"query": {"geosearch": []}}
            articles = self.fetcher.get_nearby_articles(0.0, 0.0)
            assert isinstance(articles, list)
            assert len(articles) == 0
