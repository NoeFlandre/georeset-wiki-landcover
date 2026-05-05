
import requests
from typing import List, Dict
import time


class WikiFetcher:
    def __init__(self):
        self.api_url = "https://fr.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "GeoResetPipeline/1.0 - Academic Research"}

    def get_nearby_articles(self, lat: float, lon: float, radius: int = 2000, retries: int = 3) -> List[Dict]:
        """
        Find Wikipedia articles within a radius in meters around the coordinates (latitude, longitude)
        """
        params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": min(radius, 10000),  # API caps at 10km
            "gslimit": 500,
            "format": "json"
        }
        for attempt in range(retries):
            try:
                response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
                response.raise_for_status()
                time.sleep(0.1)
                return response.json().get("query", {}).get("geosearch", [])
            except (requests.RequestException, ValueError):
                if attempt < retries - 1:
                    time.sleep(1)
                continue
        return []

    def get_articles_in_bbox(self, north: float, west: float, south: float, east: float, retries: int = 3) -> List[Dict]:
        """
        Find Wikipedia articles within a rectangular bounding box.
        Coordinates order: Top(North), Left(West), Bottom(South), Right(East).
        Handles pagination automatically when results exceed 500.
        """
        all_articles = []
        continuation = None

        while True:
            params = {
                "action": "query",
                "list": "geosearch",
                "gsbbox": f"{north}|{west}|{south}|{east}",
                "gslimit": 500,
                "format": "json"
            }
            if continuation:
                params["gscontinue"] = continuation

            for attempt in range(retries):
                try:
                    response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
                    response.raise_for_status()
                    time.sleep(0.1)
                    data = response.json()
                    break
                except (requests.RequestException, ValueError):
                    if attempt < retries - 1:
                        time.sleep(1)
                    continue
            else:
                break  # Failed all retries

            geosearch = data.get("query", {}).get("geosearch", [])
            all_articles.extend(geosearch)

            # Handle continuation
            continue_info = data.get("continue", {})
            continuation = continue_info.get("gscontinue")
            if not continuation:
                break

        return all_articles

    def get_articles_in_bounds(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float, polygon_filter=None) -> List[Dict]:
        """
        Fetch Wikipedia articles within a bounding box by tiling small bboxes.
        API has limits on bbox size, so we split into tiles and merge results.
        If polygon_filter is provided (callable that takes lon, lat and returns bool),
        only returns articles within polygons.
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

                for article in self.get_articles_in_bbox(north=north, west=west, south=lat, east=east):
                    if article["pageid"] not in seen_ids:
                        # Filter to only articles within our precise bounds
                        if self._in_bounds(article, min_lon, min_lat, max_lon, max_lat):
                            # If polygon_filter provided, check if article is within any polygon
                            if polygon_filter and not polygon_filter(article["lon"], article["lat"]):
                                continue
                            seen_ids.add(article["pageid"])
                            articles.append(article)

                lon += lon_step
            lat += lat_step

        print(f"Made {tile_count} API calls, found {len(articles)} unique articles")

        return articles

    def _in_bounds(self, article: Dict, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
        """Check if article coordinates fall within bounds."""
        return min_lon <= article["lon"] <= max_lon and min_lat <= article["lat"] <= max_lat


if __name__ == "__main__":
    import json
    from src.data_fetcher import DataFetcher

    fetcher = WikiFetcher()

    with open("data/bounds.json") as f:
        bounds = json.load(f)

    print("Fetching Wikipedia articles for Alsace region...")

    # Load polygons for filtering
    data_fetcher = DataFetcher()
    gdf = data_fetcher.load_data()

    # Create polygon filter function
    def polygon_filter(lon, lat):
        from shapely.geometry import Point
        point = Point(lon, lat)
        return any(gdf.geometry.contains(point))

    articles = fetcher.get_articles_in_bounds(
        bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"],
        polygon_filter=polygon_filter
    )
    print(f"Found {len(articles)} unique articles")

    # Add URL to each article
    for article in articles:
        from urllib.parse import quote
        article["url"] = f"https://fr.wikipedia.org/wiki/{quote(article['title'])}"

    output_path = "data/wiki_articles.json"
    with open(output_path, "w") as f:
        json.dump(articles, f, indent=2)
    print(f"Saved to {output_path}")