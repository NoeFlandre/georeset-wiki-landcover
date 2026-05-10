"""Fetch polygon features from OpenStreetMap within a bounding box."""

import logging
import time
from collections.abc import Iterator
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

LANDUSE_VALUES = (
    "farmland",
    "farmyard",
    "meadow",
    "orchard",
    "vineyard",
    "forest",
    "allotments",
    "plant_nursery",
    "greenhouse_horticulture",
    "grass",
)
NATURAL_VALUES = (
    "wood",
    "scrub",
    "grassland",
    "wetland",
    "heath",
    "water",
    "bare_rock",
    "sand",
    "scree",
    "shingle",
    "beach",
    "mud",
)


class OSMFetchError(RuntimeError):
    """Raised when an Overpass API request cannot be completed."""


class OSMFetcher:
    """Fetch closed OSM ways from Overpass as polygons."""

    def __init__(
        self,
        api_url: str = "https://overpass-api.de/api/interpreter",
        tile_size: float = 10,
        retries: int = 3,
    ):
        self.api_url = api_url
        self.tile_size = tile_size
        self.retries = retries
        self.headers = {"User-Agent": "GeoResetPipeline/1.0"}

    def fetch_polygons(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> gpd.GeoDataFrame:
        """Return OSM polygon ways inside bounds matching CORINE order."""
        if min_lon >= max_lon or min_lat >= max_lat:
            return self._empty_gdf()
        frames = [
            self._fetch_tile(west, south, east, north)
            for west, south, east, north in self._tiles(min_lon, min_lat, max_lon, max_lat)
        ]
        rows = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        if rows.empty:
            return self._empty_gdf()
        return rows.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)

    def _fetch_tile(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> gpd.GeoDataFrame:
        last_error: Exception | None = None

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.api_url,
                    data={"data": self._query(min_lon, min_lat, max_lon, max_lat)},
                    headers=self.headers,
                    timeout=180,
                )
                if self._is_transient_status(response.status_code):
                    last_error = requests.HTTPError(
                        f"Transient Overpass HTTP status {response.status_code}"
                    )
                    if attempt < self.retries - 1:
                        self._sleep_before_retry(response.status_code, attempt)
                        continue
                    break

                response.raise_for_status()
                return self._elements_to_gdf(response.json().get("elements", []))
            except requests.HTTPError as exc:
                raise OSMFetchError(
                    "Overpass request failed with non-transient HTTP error"
                ) from exc
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    self._sleep_before_retry(type(exc).__name__, attempt)
                    continue

        raise OSMFetchError(
            f"Failed to fetch OSM tile after {self.retries} retries"
        ) from last_error

    def _is_transient_status(self, status_code: int) -> bool:
        if not isinstance(status_code, int):
            return False
        return status_code == 429 or 500 <= status_code < 600

    def _sleep_before_retry(self, reason: object, attempt: int) -> None:
        wait_time = 2**attempt
        logger.info("Overpass transient failure (%s). Retrying in %ss", reason, wait_time)
        time.sleep(wait_time)

    def _tiles(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> Iterator[tuple[float, float, float, float]]:
        south = min_lat
        while south < max_lat:
            north = min(south + self.tile_size, max_lat)
            west = min_lon
            while west < max_lon:
                east = min(west + self.tile_size, max_lon)
                yield west, south, east, north
                west = east
            south = north

    def _query(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
        bbox = f"({min_lat},{min_lon},{max_lat},{max_lon})"
        landuse_values = "|".join(LANDUSE_VALUES)
        natural_values = "|".join(NATURAL_VALUES)
        return f"""
        [out:json][timeout:180];
        (
          way["area"!~"no"]["landuse"~"^({landuse_values})$"]{bbox};
          way["area"!~"no"]["natural"~"^({natural_values})$"]{bbox};
        );
        out geom;
        """

    def _elements_to_gdf(self, elements: list[dict[str, Any]]) -> gpd.GeoDataFrame:
        rows = []
        for element in elements:
            geometry = self._polygon_from_element(element)
            if geometry is None:
                continue

            tags = element.get("tags", {})
            rows.append(
                {
                    "osm_id": f"{element.get('type')}/{element.get('id')}",
                    "name": tags.get("name"),
                    "landuse": tags.get("landuse"),
                    "natural": tags.get("natural"),
                    "leisure": tags.get("leisure"),
                    "amenity": tags.get("amenity"),
                    "building": tags.get("building"),
                    "geometry": geometry,
                }
            )

        if not rows:
            return self._empty_gdf()
        return gpd.GeoDataFrame(rows, columns=self._columns(), geometry="geometry", crs="EPSG:4326")

    def _polygon_from_element(self, element: dict[str, Any]) -> Polygon | None:
        coords = [(point["lon"], point["lat"]) for point in element.get("geometry", [])]
        if len(coords) < 4 or coords[0] != coords[-1]:
            return None

        polygon = Polygon(coords)
        if polygon.is_empty or not polygon.is_valid:
            return None
        return polygon

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame([], columns=self._columns(), geometry="geometry", crs="EPSG:4326")

    def _columns(self) -> list[str]:
        return [
            "osm_id",
            "name",
            "landuse",
            "natural",
            "leisure",
            "amenity",
            "building",
            "geometry",
        ]
