from dataclasses import FrozenInstanceError

import pytest

from georeset_wiki_landcover.config import DataPaths, ModelSettings


def test_data_paths_preserve_existing_defaults():
    paths = DataPaths()

    assert paths.wiki_articles == "data/wiki/wiki_articles.json"
    assert paths.article_contents == "data/wiki/article_contents.json"
    assert paths.article_summaries == "data/wiki/article_summaries.json"
    assert paths.article_summaries_no_place == "data/wiki/article_summaries_no_place.json"
    assert paths.osm_polygons == "data/osm/osm_project_polygons.geojson"
    assert (
        paths.corine_polygons == "data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp"
    )
    assert paths.corine_bounds == "data/corine/bounds.json"
    assert paths.classification_output_dir == "data/classification/runs/default"
    assert paths.distribution_csv == "data/distribution/osm_corine_distribution.csv"
    assert paths.map_articles == "data/maps/corine_with_articles.html"
    assert paths.map_osm == "data/maps/osm_corine_polygons.html"
    assert paths.map_default == "data/maps/map.html"


def test_model_settings_preserve_existing_defaults(monkeypatch):
    monkeypatch.delenv("GEORESET_WIKI_LANDCOVER_MODEL_PATH", raising=False)

    settings = ModelSettings.from_env()

    assert settings.model_path == "Qwen3.6-27B-Q4_0.gguf"
    assert settings.seed == 42
    assert settings.classification_temperature == 0.0
    assert settings.summarization_temperature == 0.7


def test_model_settings_respect_env_override(monkeypatch):
    monkeypatch.setenv("GEORESET_WIKI_LANDCOVER_MODEL_PATH", "custom.gguf")

    assert ModelSettings.from_env().model_path == "custom.gguf"


def test_config_dataclasses_are_frozen():
    paths = DataPaths()

    with pytest.raises(FrozenInstanceError):
        paths.wiki_articles = "other.json"
