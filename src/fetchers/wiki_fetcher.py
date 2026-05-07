import time

import requests


class WikiFetchError(RuntimeError):
    """Raised when a Wikipedia API request cannot be completed."""


class WikiFetcher:
    def __init__(self):
        self.api_url = "https://fr.wikipedia.org/w/api.php"
        self.headers = {
            "User-Agent": "GeoResetPipeline/1.0 (https://geo-reset.sylvainlobry.com/; research@sylvainlobry.com) python-requests/2.33.1"
        }

    def get_nearby_articles(
        self, lat: float, lon: float, radius: int = 2000, retries: int = 3
    ) -> list[dict]:
        """
        Find Wikipedia articles within a radius in meters around the coordinates (latitude, longitude)
        """
        params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": min(radius, 10000),  # API caps at 10km
            "gslimit": 500,
            "format": "json",
        }
        for attempt in range(retries):
            try:
                response = requests.get(
                    self.api_url, params=params, headers=self.headers, timeout=10
                )
                response.raise_for_status()
                time.sleep(0.1)
                return response.json().get("query", {}).get("geosearch", [])  # type: ignore[no-any-return]
            except (requests.RequestException, ValueError):
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        return []

    def get_articles_in_bbox(
        self, north: float, west: float, south: float, east: float, retries: int = 3
    ) -> list[dict]:
        """
        Find Wikipedia articles within a rectangular bounding box.
        Coordinates order: Top(North), Left(West), Bottom(South), Right(East).
        Handles pagination automatically when results exceed 500.
        Raises WikiFetchError if any page cannot be fetched after retries.
        """
        all_articles = []
        continuation = None

        while True:
            params = {
                "action": "query",
                "list": "geosearch",
                "gsbbox": f"{north}|{west}|{south}|{east}",
                "gslimit": 500,
                "format": "json",
            }
            if continuation:
                params["gscontinue"] = continuation

            data = self._request_json(params, retries, timeout=15)

            geosearch = data.get("query", {}).get("geosearch", [])
            all_articles.extend(geosearch)

            # Handle continuation
            continue_info = data.get("continue", {})
            continuation = continue_info.get("gscontinue")
            if not continuation:
                break

        return all_articles

    def get_articles_in_bounds(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        polygon_filter=None,
        osm_polygon_filter=None,
    ) -> list[dict]:
        """
        Fetch Wikipedia articles within a bounding box by tiling small bboxes.
        API has limits on bbox size, so we split into tiles and merge results.
        If polygon_filter is provided (callable that takes lon, lat and returns bool),
        only returns articles within corine polygons.
        If osm_polygon_filter is provided, only returns articles within OSM polygons.
        An article is kept if it is in a corine polygon OR in an OSM polygon (or both).
        If only one filter is provided, articles must satisfy that filter.
        """
        articles = []
        seen_ids = set()

        # Tile the area - API works with ~0.2 degree tiles
        lat_step = 0.18  # degrees
        lon_step = 0.26  # degrees (at latitude 48°, this is ~20km)

        tile_count = 0
        lat = min_lat
        while lat < max_lat:
            lon = min_lon
            north = min(lat + lat_step, max_lat)
            while lon < max_lon:
                west = lon
                east = min(lon + lon_step, max_lon)
                tile_count += 1

                tile_articles = self.get_articles_in_bbox(
                    north=north, west=west, south=lat, east=east
                )
                print(
                    f"Tile {tile_count}: ({lat:.4f}, {lon:.4f}) to ({north:.4f}, {east:.4f}) -> {len(tile_articles)} articles"
                )
                for article in tile_articles:
                    pageid = article.get("pageid")
                    if (
                        pageid is not None
                        and pageid not in seen_ids
                        and self._in_bounds(article, min_lon, min_lat, max_lon, max_lat)
                    ):
                        # Determine if article is in corine and/or OSM
                        in_corine = polygon_filter is None or polygon_filter(
                            article["lon"], article["lat"]
                        )
                        in_osm = osm_polygon_filter is None or osm_polygon_filter(
                            article["lon"], article["lat"]
                        )

                        # If both filters provided, keep if in either
                        # If only one filter provided, article must satisfy it
                        keep = False
                        if polygon_filter is not None and osm_polygon_filter is not None:
                            keep = in_corine or in_osm
                        elif polygon_filter is not None:
                            keep = in_corine
                        elif osm_polygon_filter is not None:
                            keep = in_osm
                        else:
                            keep = True

                        if not keep:
                            continue
                        seen_ids.add(pageid)
                        articles.append(article)

                lon += lon_step
            lat += lat_step

        print(f"Made {tile_count} API calls, found {len(articles)} unique articles")

        return articles

    def _in_bounds(
        self, article: dict, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> bool:
        """Check if article coordinates fall within bounds."""
        lon = article.get("lon")
        lat = article.get("lat")
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            return False
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

    def _request_json(self, params: dict, retries: int, timeout: int) -> dict:
        """Fetch one Wikipedia API page, retrying transient failures."""
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                response = requests.get(
                    self.api_url, params=params, headers=self.headers, timeout=timeout
                )

                if response.status_code == 429:
                    self._sleep_before_retry("Rate limited (429)", attempt, scale=5)
                    continue

                response.raise_for_status()

                if "application/json" not in response.headers.get("Content-Type", ""):
                    if "too many requests" in response.text.lower():
                        self._sleep_before_retry("Rate limited (HTML response)", attempt, scale=10)
                        continue
                    raise ValueError("Non-JSON response received")

                data = response.json()

                error_code = data.get("error", {}).get("code")
                if error_code in {"ratelimited", "request-too-large"}:
                    self._sleep_before_retry(f"API Error ({error_code})", attempt, scale=5)
                    continue

                time.sleep(0.5)
                return data  # type: ignore[no-any-return]
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    self._sleep_before_retry(
                        f"Request failed ({type(exc).__name__})", attempt, scale=2
                    )

        raise WikiFetchError(
            f"Failed to fetch Wikipedia API page after {retries} retries"
        ) from last_error

    def _sleep_before_retry(self, message: str, attempt: int, scale: int) -> None:
        wait_time = (2**attempt) * scale
        print(f"  {message}. Retrying in {wait_time}s...")
        time.sleep(wait_time)


if __name__ == "__main__":
    import json

    from src.fetchers.data_fetcher import DataFetcher

    fetcher = WikiFetcher()

    with open("data/corine/bounds.json") as f:
        bounds = json.load(f)

    print("Fetching Wikipedia articles for Alsace region...")

    # Load polygons for filtering
    data_fetcher = DataFetcher()
    gdf = data_fetcher.load_data()

    # Create polygon filter function for CORINE
    def polygon_filter(lon, lat):
        from shapely.geometry import Point

        point = Point(lon, lat)
        return any(gdf.geometry.contains(point))

    # Load OSM polygons for filtering
    import geopandas as gpd

    osm_gdf = gpd.read_file("data/osm/osm_project_polygons.geojson")

    # Create polygon filter function for OSM
    def osm_polygon_filter(lon, lat):
        from shapely.geometry import Point

        point = Point(lon, lat)
        return any(osm_gdf.geometry.contains(point))

    articles = fetcher.get_articles_in_bounds(
        bounds["min_lon"],
        bounds["min_lat"],
        bounds["max_lon"],
        bounds["max_lat"],
        polygon_filter=polygon_filter,
        osm_polygon_filter=osm_polygon_filter,
    )
    print(f"Found {len(articles)} unique articles")

    # Add URL to each article
    for article in articles:
        from urllib.parse import quote

        article["url"] = f"https://fr.wikipedia.org/wiki/{quote(article['title'])}"

    output_path = "data/wiki/wiki_articles.json"
    with open(output_path, "w") as f:
        json.dump(articles, f, indent=2)
    print(f"Saved to {output_path}")
