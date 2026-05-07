"""Tests for fetching OSM polygons inside CORINE bounds."""

from unittest.mock import patch

from shapely.geometry import Polygon

from src.fetchers.osm_fetcher import OSMFetcher


def test_fetch_polygons_uses_corine_bounds_in_overpass_order():
    fetcher = OSMFetcher(tile_size=10)

    with patch("src.fetchers.osm_fetcher.requests.post") as post:
        post.return_value.json.return_value = {"elements": []}
        post.return_value.raise_for_status.return_value = None

        fetcher.fetch_polygons(min_lon=7.0, min_lat=48.0, max_lon=8.0, max_lat=49.0)

    query = post.call_args.kwargs["data"]["data"]
    assert "(48.0,7.0,49.0,8.0)" in query
    assert "User-Agent" in post.call_args.kwargs["headers"]
    assert "farmland|farmyard|meadow|orchard|vineyard|forest" in query
    assert '"building"' not in query
    assert '"residential"' not in query


def test_fetch_polygons_returns_closed_way_polygons():
    fetcher = OSMFetcher()
    response = {
        "elements": [
            {
                "type": "way",
                "id": 12,
                "tags": {"landuse": "forest", "name": "Forest"},
                "geometry": [
                    {"lat": 48.0, "lon": 7.0},
                    {"lat": 48.0, "lon": 7.1},
                    {"lat": 48.1, "lon": 7.1},
                    {"lat": 48.1, "lon": 7.0},
                    {"lat": 48.0, "lon": 7.0},
                ],
            }
        ]
    }

    with patch("src.fetchers.osm_fetcher.requests.post") as post:
        post.return_value.json.return_value = response
        post.return_value.raise_for_status.return_value = None

        gdf = fetcher.fetch_polygons(min_lon=7.0, min_lat=48.0, max_lon=8.0, max_lat=49.0)

    assert len(gdf) == 1
    assert gdf.loc[0, "osm_id"] == "way/12"
    assert gdf.loc[0, "landuse"] == "forest"
    assert isinstance(gdf.loc[0, "geometry"], Polygon)
    assert gdf.crs == "EPSG:4326"


def test_fetch_polygons_skips_open_ways():
    fetcher = OSMFetcher()
    response = {
        "elements": [
            {
                "type": "way",
                "id": 12,
                "geometry": [
                    {"lat": 48.0, "lon": 7.0},
                    {"lat": 48.0, "lon": 7.1},
                    {"lat": 48.1, "lon": 7.1},
                ],
            }
        ]
    }

    with patch("src.fetchers.osm_fetcher.requests.post") as post:
        post.return_value.json.return_value = response
        post.return_value.raise_for_status.return_value = None

        gdf = fetcher.fetch_polygons(min_lon=7.0, min_lat=48.0, max_lon=8.0, max_lat=49.0)

    assert gdf.empty


def test_fetch_polygons_tiles_large_bounds():
    fetcher = OSMFetcher(tile_size=0.5)

    with patch("src.fetchers.osm_fetcher.requests.post") as post:
        post.return_value.json.return_value = {"elements": []}
        post.return_value.raise_for_status.return_value = None

        fetcher.fetch_polygons(min_lon=7.0, min_lat=48.0, max_lon=8.2, max_lat=49.2)

    assert post.call_count == 9


def test_fetch_polygons_retries_rate_limited_tiles():
    fetcher = OSMFetcher(tile_size=10)

    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests

                raise requests.HTTPError("http error")

        def json(self):
            return self._payload

    with (
        patch("src.fetchers.osm_fetcher.time.sleep") as sleep,
        patch("src.fetchers.osm_fetcher.requests.post") as post,
    ):
        post.side_effect = [
            Response(429, {}),
            Response(200, {"elements": []}),
        ]

        gdf = fetcher.fetch_polygons(min_lon=7.0, min_lat=48.0, max_lon=8.0, max_lat=49.0)

    assert gdf.empty
    assert post.call_count == 2
    sleep.assert_called_once()
