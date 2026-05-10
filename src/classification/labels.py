from collections.abc import Mapping
from typing import Any

import geopandas as gpd

from src.fetchers.osm_fetcher import LANDUSE_VALUES, NATURAL_VALUES

CORINE_LEVEL2_DESCRIPTIONS = {
    "21": "Arable land",
    "22": "Permanent crops",
    "23": "Pastures",
    "24": "Heterogeneous agricultural areas",
    "31": "Forests",
    "32": "Shrub and/or herbaceous vegetation associations",
    "33": "Open spaces with little or no vegetation",
    "41": "Inland wetlands",
    "51": "Inland waters",
}


def corine_level2_labels(corine_gdf: gpd.GeoDataFrame) -> list[str]:
    codes = corine_gdf["code_18"].astype(str)
    return sorted({code[:2] for code in codes if not code.startswith("1")})


def osm_allowed_labels() -> list[str]:
    return sorted(set(LANDUSE_VALUES) | set(NATURAL_VALUES))


def osm_labels_from_row(row: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    landuse = row.get("landuse")
    natural = row.get("natural")
    if landuse in LANDUSE_VALUES:
        labels.append(str(landuse))
    if natural in NATURAL_VALUES and str(natural) not in labels:
        labels.append(str(natural))
    return labels


def osm_label_from_row(row: Mapping[str, Any]) -> str | None:
    labels = osm_labels_from_row(row)
    return labels[0] if labels else None
