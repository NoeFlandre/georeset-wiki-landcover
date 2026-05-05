
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

    def get_articles_in_bounds(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float, radius: int = 10000) -> List[Dict]:
        """
        Fetch Wikipedia articles within a bounding box by sampling a grid of points
        and filtering to only those inside the bounds.

        Uses a grid of circles spaced by diameter to cover the entire bounds area.
        """
        articles = []
        seen_ids = set()

        # Grid spacing based on circle diameter (circles just touch at edges)
        # At latitude 48°, 1° ≈ 74km (lon) and 111km (lat)
        # Coverage verified by test_grid_coverage_is_complete
        km_per_deg_lon = 74
        km_per_deg_lat = 111
        step_lon = (radius / 1000) / km_per_deg_lon
        step_lat = (radius / 1000) / km_per_deg_lat

        # Count total calls needed
        lon = min_lon
        total_lons = 0
        while lon <= max_lon:
            total_lons += 1
            lon += step_lon
        lat = min_lat
        total_lats = 0
        while lat <= max_lat:
            total_lats += 1
            lat += step_lat
        total_calls = total_lons * total_lats
        print(f"Grid: {total_lons} x {total_lats} = {total_calls} API calls")

        lon = min_lon
        call_num = 0
        while lon <= max_lon:
            lat = min_lat
            while lat <= max_lat:
                call_num += 1
                if call_num % 10 == 0:
                    print(f"  Call {call_num}/{total_calls}...")
                for article in self.get_nearby_articles(lat, lon, radius):
                    if article["pageid"] not in seen_ids and self._in_bounds(article, min_lon, min_lat, max_lon, max_lat):
                        seen_ids.add(article["pageid"])
                        articles.append(article)
                lat += step_lat
            lon += step_lon

        return articles

    def _in_bounds(self, article: Dict, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
        """Check if article coordinates fall within bounds."""
        return min_lon <= article["lon"] <= max_lon and min_lat <= article["lat"] <= max_lat


if __name__ == "__main__":
    import json

    fetcher = WikiFetcher()

    with open("data/bounds.json") as f:
        bounds = json.load(f)

    print(f"Fetching Wikipedia articles for Alsace region...")
    articles = fetcher.get_articles_in_bounds(
        bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"]
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