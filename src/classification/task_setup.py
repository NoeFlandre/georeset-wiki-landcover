"""Task-specific ground-truth and label setup for classification runs."""

from dataclasses import dataclass
from typing import Any

import geopandas as gpd

from src.classification.ground_truth import build_corine_ground_truth, build_osm_ground_truth
from src.classification.labels import (
    CORINE_LEVEL2_DESCRIPTIONS,
    corine_level2_labels,
    osm_allowed_labels,
)
from src.fetchers.data_fetcher import DataFetcher


@dataclass(frozen=True)
class ClassificationTaskSetup:
    target: dict[str, Any]
    allowed_labels: list[str]
    label_descriptions: dict[str, str]


def load_task_setup(
    *,
    task: str,
    articles: list[dict],
    corine_polygons_path: str,
    osm_polygons_path: str,
) -> ClassificationTaskSetup:
    if task == "corine_level2":
        corine_gdf = DataFetcher(corine_polygons_path).load_data(exclude_artificial=True)
        allowed_labels = corine_level2_labels(corine_gdf)
        return ClassificationTaskSetup(
            target=build_corine_ground_truth(articles, corine_gdf),
            allowed_labels=allowed_labels,
            label_descriptions={
                label: description
                for label, description in CORINE_LEVEL2_DESCRIPTIONS.items()
                if label in allowed_labels
            },
        )
    if task == "osm":
        osm_gdf = gpd.read_file(osm_polygons_path)
        return ClassificationTaskSetup(
            target=build_osm_ground_truth(articles, osm_gdf),
            allowed_labels=osm_allowed_labels(),
            label_descriptions={},
        )
    raise ValueError(f"Unknown classification task: {task}")
