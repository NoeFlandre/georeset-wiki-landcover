import json

from georeset_wiki_landcover.text.retrieved_evidence_windows import (
    build_retrieved_evidence_window_record,
)


def test_retrieved_windows_selects_matching_evidence_sentence_with_context() -> None:
    record = build_retrieved_evidence_window_record(
        pageid="100",
        article={
            "title": "Lieu Test",
            "content": (
                "Introduction sans indice. "
                "La vallée comprend des prairies humides. "
                "Des forêts entourent le site. "
                "Conclusion administrative."
            ),
        },
        evidence={
            "evidence_sentences_no_place": ["Des forêts entourent le site."],
            "landcover_relevance": "high",
            "uncertainty": "low",
            "evidence_types": ["forest"],
        },
        seed=42,
    )

    retrieved = record["retrieved_evidence_windows"]

    assert "Phrases brutes extraites de l'article." in retrieved
    assert "La vallée comprend des prairies humides." in retrieved
    assert "Des forêts entourent le site." in retrieved
    assert "Conclusion administrative." in retrieved
    assert "Introduction sans indice." not in retrieved
    assert record["retrieved_sentence_count"] == 3
    assert record["matched_evidence_sentence_count"] == 1
    assert record["metadata"]["source"] == "deterministic_retrieved_evidence_windows"
    assert record["metadata"]["uses_raw_content"] is True


def test_sentence_only_variant_omits_neighboring_context() -> None:
    record = build_retrieved_evidence_window_record(
        pageid="100",
        article={
            "title": "Lieu",
            "content": "Avant. La zone est boisée. Après.",
        },
        evidence={"evidence_sentences_no_place": ["La zone est boisée."]},
        seed=42,
    )

    assert "La zone est boisée." in record["retrieved_evidence_sentences_only"]
    assert "Avant." not in record["retrieved_evidence_sentences_only"]
    assert "Après." not in record["retrieved_evidence_sentences_only"]


def test_retrieved_windows_use_evidence_type_keywords_when_sentence_text_was_masked() -> None:
    record = build_retrieved_evidence_window_record(
        pageid="100",
        article={
            "title": "Lieu",
            "content": "Phrase historique. Une forêt dense couvre le versant. Phrase finale.",
        },
        evidence={
            "evidence_sentences_no_place": ["ce lieu possède des boisements."],
            "evidence_types": ["forest"],
        },
        seed=42,
    )

    assert "Une forêt dense couvre le versant." in record["retrieved_evidence_windows"]
    assert record["matched_evidence_sentence_count"] == 1


def test_random_windows_are_deterministic_and_use_same_sentence_budget() -> None:
    article = {
        "title": "Lieu",
        "content": "Un. Deux. Trois. Quatre. Cinq.",
    }
    evidence = {"evidence_sentences_no_place": ["Trois."]}

    first = build_retrieved_evidence_window_record(
        pageid="100", article=article, evidence=evidence, seed=7
    )
    second = build_retrieved_evidence_window_record(
        pageid="100", article=article, evidence=evidence, seed=7
    )

    assert first["random_sentence_windows"] == second["random_sentence_windows"]
    assert first["random_sentence_count"] == first["retrieved_sentence_count"]


def test_no_place_variant_masks_title_only_in_retrieved_windows() -> None:
    record = build_retrieved_evidence_window_record(
        pageid="100",
        article={
            "title": "Forêt de Test",
            "content": "Forêt de Test possède des boisements. Une rivière traverse Forêt de Test.",
        },
        evidence={"evidence_sentences_no_place": ["ce lieu possède des boisements."]},
        seed=42,
    )

    assert "Forêt de Test" in record["retrieved_evidence_windows"]
    assert "Forêt de Test" not in record["retrieved_evidence_windows_no_place"]
    assert "ce lieu possède des boisements" in record["retrieved_evidence_windows_no_place"]
    json.dumps(record, allow_nan=False)
