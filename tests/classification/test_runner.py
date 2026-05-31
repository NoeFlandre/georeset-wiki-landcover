import json

import pytest

from georeset_wiki_landcover.classification.runner import (
    compute_metrics,
    load_text_source,
    main,
    parse_args,
    prediction_fingerprint,
    validate_prediction_result,
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


def test_compute_metrics_rejects_empty_label_universe():
    with pytest.raises(
        ValueError,
        match="No labels available for task=osm text_source=summary",
    ):
        compute_metrics("osm", "summary", {"1": []}, {"1": []}, [])


def test_main_rejects_runs_with_no_eligible_records(tmp_path):
    wiki_path = tmp_path / "wiki.json"
    contents_path = tmp_path / "contents.json"
    summaries_path = tmp_path / "summaries.json"
    no_place_path = tmp_path / "no_place.json"
    corine_path = tmp_path / "corine.geojson"
    output_dir = tmp_path / "out"

    wiki_path.write_text(
        json.dumps([{"pageid": 100, "lat": 10.0, "lon": 10.0, "title": "Outside"}]),
        encoding="utf-8",
    )
    contents_path.write_text(json.dumps({"100": {"content": "Outside"}}), encoding="utf-8")
    summaries_path.write_text(json.dumps({"100": {"summary": "Outside"}}), encoding="utf-8")
    no_place_path.write_text(json.dumps({"100": {"summary": "Outside"}}), encoding="utf-8")

    import geopandas as gpd
    from shapely.geometry import box

    corine = gpd.GeoDataFrame(
        {"code_18": ["311"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    corine.to_file(corine_path, driver="GeoJSON")

    def classifier_factory(*args, **kwargs):
        raise AssertionError("classifier should not be constructed when no records are eligible")

    with pytest.raises(
        ValueError,
        match="No eligible records for task=corine_level2 text_source=summary",
    ):
        main(
            [
                "--task",
                "corine_level2",
                "--text-source",
                "summary",
                "--wiki-articles-path",
                str(wiki_path),
                "--article-contents-path",
                str(contents_path),
                "--article-summaries-path",
                str(summaries_path),
                "--article-summaries-no-place-path",
                str(no_place_path),
                "--corine-polygons-path",
                str(corine_path),
                "--output-dir",
                str(output_dir),
                "--model-path",
                "m.gguf",
            ],
            classifier_factory=classifier_factory,
        )


def test_validate_prediction_result_rejects_malformed_classifier_result():
    with pytest.raises(ValueError, match="pageid 100 produced invalid parse_status"):
        validate_prediction_result(
            {
                "prediction": "31",
                "prediction_labels": ["31"],
                "parse_status": "complete",
                "raw_response": "{}",
                "error": None,
                "metadata": {},
            },
            pageid="100",
            task="corine_level2",
            allowed_labels=["31"],
        )


def test_validate_prediction_result_rejects_unknown_prediction_labels():
    with pytest.raises(ValueError, match="pageid 100 produced unknown prediction labels: 99"):
        validate_prediction_result(
            {
                "prediction": "99",
                "prediction_labels": ["99"],
                "parse_status": "ok",
                "raw_response": '{"label": "99"}',
                "error": None,
                "metadata": {},
            },
            pageid="100",
            task="corine_level2",
            allowed_labels=["31"],
        )


def test_validate_prediction_result_rejects_non_string_prediction_labels():
    with pytest.raises(ValueError, match="pageid 100 produced non-string prediction_labels"):
        validate_prediction_result(
            {
                "prediction": "31",
                "prediction_labels": ["31", 31],
                "parse_status": "ok",
                "raw_response": '{"label": "31"}',
                "error": None,
                "metadata": {},
            },
            pageid="100",
            task="corine_level2",
            allowed_labels=["31"],
        )


def test_validate_prediction_result_rejects_ok_without_prediction():
    with pytest.raises(
        ValueError, match="pageid 100 produced ok parse_status without a prediction"
    ):
        validate_prediction_result(
            {
                "prediction": None,
                "prediction_labels": None,
                "parse_status": "ok",
                "raw_response": "{}",
                "error": None,
                "metadata": {},
            },
            pageid="100",
            task="osm",
            allowed_labels=["wood"],
        )


def test_validate_prediction_result_rejects_single_label_prediction_mismatch():
    with pytest.raises(
        ValueError,
        match="pageid 100 produced invalid single-label prediction for corine_level2",
    ):
        validate_prediction_result(
            {
                "prediction": "31",
                "prediction_labels": ["31", "32"],
                "parse_status": "ok",
                "raw_response": '{"labels": ["31", "32"]}',
                "error": None,
                "metadata": {},
            },
            pageid="100",
            task="corine_level2",
            allowed_labels=["31", "32"],
        )


def test_validate_prediction_result_rejects_non_object_metadata():
    with pytest.raises(ValueError, match="pageid 100 produced non-object metadata"):
        validate_prediction_result(
            {
                "prediction": "wood",
                "prediction_labels": ["wood"],
                "parse_status": "ok",
                "raw_response": '{"labels": ["wood"]}',
                "error": None,
                "metadata": "bad",
            },
            pageid="100",
            task="osm",
            allowed_labels=["wood"],
        )
