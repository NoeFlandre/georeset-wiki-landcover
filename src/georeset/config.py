"""Frozen default configuration for data paths and model settings."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataPaths:
    wiki_articles: str = "data/wiki/wiki_articles.json"
    article_contents: str = "data/wiki/article_contents.json"
    article_summaries: str = "data/wiki/article_summaries.json"
    article_summaries_no_place: str = "data/wiki/article_summaries_no_place.json"
    article_landuse_evidence_summaries: str = "data/wiki/article_landuse_evidence_summaries.json"
    article_evidence_cards: str = "data/wiki/article_evidence_cards.json"
    osm_polygons: str = "data/osm/osm_project_polygons.geojson"
    corine_polygons: str = "data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp"
    corine_bounds: str = "data/corine/bounds.json"
    classification_output_dir: str = "data/classification/runs/default"
    distribution_csv: str = "data/distribution/osm_corine_distribution.csv"
    map_articles: str = "data/maps/corine_with_articles.html"
    map_osm: str = "data/maps/osm_corine_polygons.html"
    map_default: str = "data/maps/map.html"


@dataclass(frozen=True)
class ModelSettings:
    model_path: str = "Qwen3.6-27B-Q4_0.gguf"
    model_repo_id: str | None = None
    seed: int = 42
    classification_temperature: float = 0.0
    summarization_temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            model_path=os.environ.get("GEORESET_MODEL_PATH", cls.model_path),
            model_repo_id=os.environ.get("GEORESET_MODEL_REPO_ID"),
        )
