
import requests
from typing import List, Dict

class WikiFetcher:
    def __init__(self):
        self.api_url = "https://fr.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "GeoResetPipeline/1.0 - Academic Research"}

    def get_nearby_articles(self, lat:float, lon:float, radius:int=2000) -> List[Dict]:
        """
        Find Wikipedia articles within a radius in meters around the coordinates (latitude, longitude)
        """

        params = {
            "action":"query",
            "list":"geosearch",
            "gscoord":f"{lat}|{lon}",
            "gsradius": radius,
            "gslimit": 5,
            "format":"json"
        }

        response = requests.get(self.api_url, params=params, headers=self.headers).json()
        return response.get("query", {}).get("geosearch", [])

if __name__ == "__main__":
    fetcher = WikiFetcher()
    articles = fetcher.get_nearby_articles(48.8, 7.7)
    
    if articles:
        print(f"Found {len(articles)} articles")

    else:
        print("No articles found in this radius")