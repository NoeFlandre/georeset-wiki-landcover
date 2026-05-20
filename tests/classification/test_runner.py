import json

import pytest

from georeset_wiki_landcover.classification.runner import (
    compute_metrics,
    load_text_source,
    parse_args,
    prediction_fingerprint,
)
from georeset_wiki_landcover.config import DataPaths, ModelSettings


def test_parse_args_uses_config_defaults(monkeypatch):
    monkeypatch.delenv("GEORESET_WIKI_LANDCOVER_MODEL_PATH", raising=False)

    args = parse_args([])

    assert args.limit is None
    assert args.wiki_articles_path == DataPaths().wiki_articles
    assert args.article_contents_path == DataPaths().article_contents
    assert args.article_summaries_path == DataPaths().article_summaries
    assert args.article_summaries_no_place_path == DataPaths().article_summaries_no_place
    assert args.article_landuse_evidence_summaries_path == (
        DataPaths().article_landuse_evidence_summaries
    )
    assert args.article_evidence_cards_path == DataPaths().article_evidence_cards
    assert args.article_evidence_highlights_path == DataPaths().article_evidence_highlights
    assert args.article_retrieved_evidence_windows_path == (
        DataPaths().article_retrieved_evidence_windows
    )
    assert args.osm_polygons_path == DataPaths().osm_polygons
    assert args.corine_polygons_path == DataPaths().corine_polygons
    assert args.output_dir == DataPaths().classification_output_dir
    assert args.model_path == ModelSettings().model_path
    assert args.model_repo_id is None
    assert args.seed == ModelSettings().seed
    assert args.temperature == ModelSettings().classification_temperature


def test_parse_args_accepts_env_model_repo_id(monkeypatch):
    monkeypatch.setenv("GEORESET_WIKI_LANDCOVER_MODEL_REPO_ID", "google/gemma-4-gguf")

    args = parse_args([])

    assert args.model_repo_id == "google/gemma-4-gguf"


def test_fingerprint_changes_when_model_repo_id_changes():
    qwen = prediction_fingerprint(
        "corine_level2", "content", "gemma.gguf", None, 42, 0.0, ["21", "31"]
    )
    gemma = prediction_fingerprint(
        "corine_level2",
        "content",
        "gemma.gguf",
        "google/gemma-4-gguf",
        42,
        0.0,
        ["21", "31"],
    )

    assert qwen != gemma


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
        str(tmp_path / "evidence.json"),
    )

    assert result == {"100": "Full text"}


def test_load_text_source_rejects_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="Unknown text source"):
        load_text_source(
            "unknown",
            "contents.json",
            "summaries.json",
            "no_place.json",
            "evidence.json",
        )


def test_load_text_source_loads_landuse_evidence_summary(tmp_path):
    summaries_path = tmp_path / "summaries.json"
    contents_path = tmp_path / "contents.json"
    no_place_path = tmp_path / "no_place.json"
    landuse_path = tmp_path / "landuse.json"
    contents_path.write_text(json.dumps({"100": {"content": "Full text"}}))
    summaries_path.write_text(json.dumps({}))
    no_place_path.write_text(json.dumps({}))
    landuse_path.write_text(json.dumps({"100": {"landuse_evidence_summary": "Land-use summary"}}))

    result = load_text_source(
        "landuse_evidence_summary",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
    )

    assert result == {"100": "Land-use summary"}


def test_load_text_source_loads_evidence_card_variants(tmp_path):
    summaries_path = tmp_path / "summaries.json"
    contents_path = tmp_path / "contents.json"
    no_place_path = tmp_path / "no_place.json"
    landuse_path = tmp_path / "landuse.json"
    evidence_cards_path = tmp_path / "cards.json"
    contents_path.write_text(json.dumps({"100": {"content": "Full text"}}))
    summaries_path.write_text(json.dumps({}))
    no_place_path.write_text(json.dumps({}))
    landuse_path.write_text(json.dumps({}))
    evidence_cards_path.write_text(
        json.dumps(
            {
                "100": {
                    "evidence_card": "Card only",
                    "content_with_evidence_card": "Card plus content",
                }
            }
        )
    )

    card = load_text_source(
        "evidence_card",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
        str(evidence_cards_path),
    )
    shuffled_card = load_text_source(
        "evidence_card_shuffled",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
        str(evidence_cards_path),
    )
    content_with_card = load_text_source(
        "content_with_evidence_card",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
        str(evidence_cards_path),
    )

    assert card == {"100": "Card only"}
    assert shuffled_card == {"100": "Card only"}
    assert content_with_card == {"100": "Card plus content"}


def test_load_text_source_loads_evidence_highlight_variants(tmp_path):
    summaries_path = tmp_path / "summaries.json"
    contents_path = tmp_path / "contents.json"
    no_place_path = tmp_path / "no_place.json"
    landuse_path = tmp_path / "landuse.json"
    evidence_cards_path = tmp_path / "cards.json"
    highlights_path = tmp_path / "highlights.json"
    contents_path.write_text(json.dumps({"100": {"content": "Full text"}}))
    summaries_path.write_text(json.dumps({}))
    no_place_path.write_text(json.dumps({}))
    landuse_path.write_text(json.dumps({}))
    evidence_cards_path.write_text(json.dumps({}))
    highlights_path.write_text(
        json.dumps({"100": {"content_with_evidence_highlights": "Highlights plus content"}})
    )

    result = load_text_source(
        "content_with_evidence_highlights_shuffled",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
        str(evidence_cards_path),
        str(highlights_path),
    )

    assert result == {"100": "Highlights plus content"}


def test_load_text_source_loads_retrieved_evidence_window_variants(tmp_path):
    summaries_path = tmp_path / "summaries.json"
    contents_path = tmp_path / "contents.json"
    no_place_path = tmp_path / "no_place.json"
    landuse_path = tmp_path / "landuse.json"
    evidence_cards_path = tmp_path / "cards.json"
    highlights_path = tmp_path / "highlights.json"
    retrieved_path = tmp_path / "retrieved.json"
    contents_path.write_text(json.dumps({"100": {"content": "Full text"}}))
    summaries_path.write_text(json.dumps({}))
    no_place_path.write_text(json.dumps({}))
    landuse_path.write_text(json.dumps({}))
    evidence_cards_path.write_text(json.dumps({}))
    highlights_path.write_text(json.dumps({}))
    retrieved_path.write_text(
        json.dumps(
            {
                "100": {
                    "retrieved_evidence_windows": "Matched windows",
                    "retrieved_evidence_sentences_only": "Matched sentences",
                    "random_sentence_windows": "Random windows",
                    "retrieved_evidence_windows_no_place": "Masked windows",
                }
            }
        )
    )

    result = load_text_source(
        "retrieved_evidence_windows_shuffled",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
        str(evidence_cards_path),
        str(highlights_path),
        str(retrieved_path),
    )
    random_result = load_text_source(
        "random_sentence_windows",
        str(contents_path),
        str(summaries_path),
        str(no_place_path),
        str(landuse_path),
        str(evidence_cards_path),
        str(highlights_path),
        str(retrieved_path),
    )

    assert result == {"100": "Matched windows"}
    assert random_result == {"100": "Random windows"}


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
