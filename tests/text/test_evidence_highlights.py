import json

from georeset_wiki_landcover.text.evidence_highlights import build_evidence_highlight_record


def test_highlighted_content_prepends_evidence_block_without_targets_or_coordinates():
    record = build_evidence_highlight_record(
        pageid="100",
        article={
            "title": "Forêt de Test",
            "content": "Le texte brut mentionne Forêt de Test, 48.5 et 7.5.",
            "lat": 48.5,
            "lon": 7.5,
        },
        evidence={
            "landcover_relevance": "high",
            "uncertainty": "low",
            "evidence_types": ["forest", "water"],
            "evidence_sentences_no_place": [
                "Des boisements sont mentionnés.",
                "Un cours d'eau traverse le paysage.",
            ],
        },
    )

    highlights = record["evidence_highlights"]
    content = record["content_with_evidence_highlights"]

    assert "Forêt de Test" not in highlights
    assert "48.5" not in highlights
    assert "7.5" not in highlights
    assert "CORINE" not in highlights
    assert "OSM" not in highlights
    assert "Des boisements sont mentionnés." in highlights
    assert "Un cours d'eau traverse le paysage." in highlights
    assert content.startswith(highlights)
    assert "\n\nTexte complet de l'article:\nLe texte brut mentionne Forêt de Test" in content
    assert record["evidence_highlights_char_count"] == len(highlights)
    assert record["content_with_evidence_highlights_char_count"] == len(content)
    assert record["evidence_sentence_count"] == 2
    assert record["metadata"]["source"] == "deterministic_evidence_highlights"
    assert record["metadata"]["version"] == 1


def test_highlighted_content_has_no_evidence_fallback_and_json_safe_values():
    record = build_evidence_highlight_record(
        pageid="100",
        article={"title": "Lieu", "content": "Texte brut."},
        evidence={
            "landcover_relevance": None,
            "uncertainty": float("nan"),
            "evidence_types": [None, "", float("nan")],
            "evidence_sentences_no_place": [],
        },
    )

    assert "Aucune phrase d'indice explicite n'est disponible." in record["evidence_highlights"]
    assert record["landcover_relevance"] is None
    assert record["uncertainty"] is None
    assert record["evidence_types"] == []
    assert json.dumps(record, allow_nan=False)
