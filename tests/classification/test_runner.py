import json

import pytest

from georeset.classification.runner import compute_metrics, load_text_source, parse_args
from georeset.config import DataPaths, ModelSettings


def test_parse_args_uses_config_defaults(monkeypatch):
    monkeypatch.delenv("GEORESET_MODEL_PATH", raising=False)

    args = parse_args([])

    assert args.limit is None
    assert args.wiki_articles_path == DataPaths().wiki_articles
    assert args.article_contents_path == DataPaths().article_contents
    assert args.article_summaries_path == DataPaths().article_summaries
    assert args.article_summaries_no_place_path == DataPaths().article_summaries_no_place
    assert args.osm_polygons_path == DataPaths().osm_polygons
    assert args.corine_polygons_path == DataPaths().corine_polygons
    assert args.output_dir == DataPaths().classification_output_dir
    assert args.model_path == ModelSettings().model_path
    assert args.seed == ModelSettings().seed
    assert args.temperature == ModelSettings().classification_temperature


def test_load_text_source_loads_content_variant(tmp_path):
    contents_path = tmp_path / "contents.json"
    summaries_path = tmp_path / "summaries.json"
    no_place_path = tmp_path / "no_place.json"
    contents_path.write_text(json.dumps({"100": {"content": "Full text"}}))
    summaries_path.write_text(json.dumps({}))
    no_place_path.write_text(json.dumps({}))

    result = load_text_source(
        "content",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
    )

    assert result == {"100": "Full text"}


def test_load_text_source_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="Unknown text source"):
        load_text_source("unknown", "contents.json", "summaries.json", "no_place.json")


def test_compute_multilabel_metrics_records_labels_evaluated():
    metrics, labels = compute_metrics(
        "osm",
        "summary",
        {"1": ["meadow", "wood"]},
        {"1": ["wood"]},
        ["meadow", "wood"],
    )

    assert labels == ["meadow", "wood"]
    assert metrics["labels_evaluated"] == ["meadow", "wood"]
    assert metrics["task"] == "osm"
