from georeset.classification.text_sources import (
    SHUFFLED_TEXT_SOURCES,
    TEXT_SOURCE_CHOICES,
    apply_shuffled_text_control,
    base_text_source,
    shuffled_metadata,
)


def test_base_text_source_maps_shuffled_variants_to_real_inputs():
    assert base_text_source("summary_shuffled") == "summary"
    assert base_text_source("summary_no_place_shuffled") == "summary_no_place"
    assert base_text_source("landuse_evidence_summary_shuffled") == "landuse_evidence_summary"
    assert base_text_source("evidence_card_shuffled") == "evidence_card"
    assert base_text_source("content_with_evidence_card_shuffled") == "content_with_evidence_card"
    assert (
        base_text_source("content_with_evidence_highlights_shuffled")
        == "content_with_evidence_highlights"
    )
    assert base_text_source("content_shuffled") == "content"
    assert base_text_source("summary") == "summary"


def test_text_source_choices_include_primary_and_shuffled_sources():
    assert TEXT_SOURCE_CHOICES == [
        "summary",
        "summary_no_place",
        "landuse_evidence_summary",
        "evidence_card",
        "content_with_evidence_card",
        "content_with_evidence_highlights",
        "content",
        "summary_shuffled",
        "summary_no_place_shuffled",
        "landuse_evidence_summary_shuffled",
        "evidence_card_shuffled",
        "content_with_evidence_card_shuffled",
        "content_with_evidence_highlights_shuffled",
        "content_shuffled",
    ]
    assert SHUFFLED_TEXT_SOURCES["content_shuffled"] == "content"
    assert SHUFFLED_TEXT_SOURCES["landuse_evidence_summary_shuffled"] == "landuse_evidence_summary"
    assert SHUFFLED_TEXT_SOURCES["evidence_card_shuffled"] == "evidence_card"
    assert (
        SHUFFLED_TEXT_SOURCES["content_with_evidence_card_shuffled"]
        == "content_with_evidence_card"
    )
    assert (
        SHUFFLED_TEXT_SOURCES["content_with_evidence_highlights_shuffled"]
        == "content_with_evidence_highlights"
    )


def test_apply_shuffled_text_control_is_deterministic_without_fixed_points():
    text_records = {"1": "Text A", "2": "Text B", "3": "Text C"}
    eligible = ["1", "2", "3"]

    shuffled_1, mapping_1 = apply_shuffled_text_control(text_records, eligible, seed=42)
    shuffled_2, mapping_2 = apply_shuffled_text_control(text_records, eligible, seed=42)

    assert shuffled_1 == shuffled_2
    assert mapping_1 == mapping_2
    assert set(mapping_1) == set(eligible)
    assert all(mapping_1[pageid] != pageid for pageid in eligible)
    assert {shuffled_1[pageid] for pageid in eligible} == {"Text A", "Text B", "Text C"}


def test_apply_shuffled_text_control_keeps_singletons_traceable():
    shuffled, mapping = apply_shuffled_text_control({"1": "Only text"}, ["1"], seed=42)

    assert shuffled == {"1": "Only text"}
    assert mapping == {"1": "1"}


def test_shuffled_metadata_describes_source_reassignment():
    metadata = shuffled_metadata("summary_shuffled", "200")

    assert metadata == {
        "text_control": "shuffled",
        "base_text_source": "summary",
        "shuffled_from_pageid": "200",
    }
