import json
import re

import pandas as pd

from georeset.fetchers.landuse_evidence_summarizer import EVIDENCE_TYPES
from georeset.text.evidence_cards import build_evidence_card_record


def test_evidence_card_excludes_title_coordinates_and_target_labels():
    record = build_evidence_card_record(
        pageid="100",
        article={
            "title": "Forêt de Test",
            "content": "Le contenu complet mentionne Forêt de Test et 48.5, 7.5.",
            "lat": 48.5,
            "lon": 7.5,
        },
        evidence={
            "landcover_relevance": "high",
            "uncertainty": "low",
            "evidence_types": ["forest", "water"],
            "evidence_sentences_no_place": ["Des boisements et un cours d'eau sont mentionnés."],
            "landuse_evidence_summary": "Le paysage comprend des boisements et de l'eau.",
        },
        article_type={"primary_article_type": "natural_landscape"},
        spatial={
            "point_label": "31",
            "dominant_label_250m": "31",
            "point_label_share_250m": 0.82,
            "point_label_share_500m": 0.71,
            "dominant_matches_point_label_250m": True,
        },
        quality={
            "quality_score": 7.2,
            "quality_bin": "quality_very_high",
            "recommended_use": "use_for_training",
        },
    )

    card = record["evidence_card"]

    assert "Forêt de Test" not in card
    assert "48.5" not in card
    assert "7.5" not in card
    assert "31" not in card
    assert "dominant_label" not in card
    assert "point_label" not in card
    assert "Pertinence: élevée" in card
    assert "Incertitude: faible" in card
    assert "Types d'indices: forêt, eau" in card
    assert "Type d'article: paysage naturel" in card
    assert "Score de qualité: 7.2" in card
    assert "Catégorie de qualité: qualité très élevée" in card
    assert "Catégorie d'usage recommandée: utilisable pour entraînement" in card
    assert "- part du label ponctuel à 250 m: 0.82" in card
    assert "- le label dominant à 250 m correspond au label ponctuel: oui" in card
    assert "- Des boisements et un cours d'eau sont mentionnés." in card


def test_evidence_card_scrubs_obvious_title_variants_from_evidence_text():
    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Forêt-de-Test", "content": ""},
        evidence={
            "evidence_sentences_no_place": ["La Forêt de Test comprend des boisements."],
            "landuse_evidence_summary": "Forêt-de-Test contient aussi des prairies.",
        },
        article_type={},
        spatial={},
        quality={},
    )

    assert "Forêt-de-Test" not in record["evidence_card"]
    assert "Forêt de Test" not in record["evidence_card"]
    assert "ce lieu comprend des boisements" in record["evidence_card"]


def test_content_with_evidence_card_prepends_card_and_preserves_raw_content():
    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Village", "content": "Texte brut avec le nom Village."},
        evidence={"landuse_evidence_summary": "Résumé sans nom."},
        article_type={},
        spatial={},
        quality={},
    )

    assert record["content_with_evidence_card"].startswith(record["evidence_card"])
    assert "\n\nTexte complet de l'article:\nTexte brut avec le nom Village." in record[
        "content_with_evidence_card"
    ]
    assert "Village" not in record["evidence_card"]
    assert "Village" in record["content_with_evidence_card"]


def test_evidence_card_uses_fallback_for_missing_sentences_and_metadata():
    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Lieu", "content": ""},
        evidence={},
        article_type={},
        spatial={},
        quality={},
    )

    card = record["evidence_card"]

    assert "Pertinence: inconnue" in card
    assert "Incertitude: inconnue" in card
    assert "Types d'indices: aucun" in card
    assert "Aucun indice factuel explicite n'a été extrait." in card
    assert record["evidence_sentence_count"] == 0
    assert record["evidence_card_char_count"] == len(card)
    assert record["content_with_evidence_card_char_count"] == len(record["content_with_evidence_card"])


def test_evidence_card_accepts_series_metadata_values():
    evidence = pd.Series(
        {
            "landcover_relevance": "medium",
            "uncertainty": "medium",
            "evidence_types": ["agriculture"],
            "evidence_sentences_count": 1,
            "landuse_evidence_summary_char_count": 42,
        }
    )

    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Lieu", "content": "Contenu."},
        evidence=evidence,
        article_type=pd.Series({"candidate_article_types": ["agriculture_or_vineyard"]}),
        spatial=pd.Series({"dominant_matches_point_label_250m": False}),
        quality=pd.Series({"quality_bin": "quality_medium"}),
    )

    assert "Pertinence: moyenne" in record["evidence_card"]
    assert "agriculture" in record["evidence_card"]
    assert "agriculture ou vignoble" in record["evidence_card"]
    assert "agriculture_or_vineyard" in record["candidate_article_types"]
    assert "non" in record["evidence_card"]


def test_evidence_card_renders_readable_metadata_labels():
    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Lieu", "content": "Texte d'exemple."},
        evidence={
            "landcover_relevance": "low",
            "uncertainty": "high",
            "evidence_types": ["urban_or_artificial", "water"],
        },
        article_type={
            "primary_article_type": "settlement_or_administrative",
            "candidate_article_types": ["settlement_or_administrative", "agriculture_or_vineyard"],
        },
        spatial={},
        quality={
            "quality_bin": "quality_medium",
            "recommended_use": "inspect_manually",
        },
    )

    card = record["evidence_card"]

    assert "Type d'article: zone urbaine / administrative" in card
    assert "Types d'article candidats: zone urbaine / administrative, agriculture ou vignoble" in card
    assert "Pertinence: faible" in card
    assert "Incertitude: élevée" in card
    assert "Types d'indices: urbain ou artificiel, eau" in card
    assert "Catégorie de qualité: qualité moyenne" in card
    assert "Catégorie d'usage recommandée: à inspecter manuellement" in card
    assert "exclude" not in card
    assert "other_or_unclear" not in card
    assert "settlement_or_administrative" not in card
    assert "use_for_training" not in card


def test_evidence_card_renders_all_fixed_evidence_types_as_french_labels():
    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Lieu", "content": "Texte d'exemple."},
        evidence={"evidence_types": list(EVIDENCE_TYPES)},
        article_type={},
        spatial={},
        quality={},
    )

    card = record["evidence_card"]

    expected_labels = [
        "forêt",
        "agriculture",
        "vignoble",
        "prairie ou pâturage",
        "eau",
        "zone humide",
        "végétation arbustive ou herbacée",
        "sol nu ou rocheux",
        "urbain ou artificiel",
        "relief ou géologie",
        "habitat ou écologie",
    ]
    for label in expected_labels:
        assert label in card

    for evidence_type in EVIDENCE_TYPES:
        if "_" in evidence_type:
            assert evidence_type not in card
            assert evidence_type.replace("_", " ") not in card


def test_evidence_card_handles_missing_scalar_and_list_values_as_json_safe():
    record = build_evidence_card_record(
        pageid="100",
        article={"title": "Lieu", "content": "Texte d'exemple."},
        evidence=pd.Series(
            {
                "landcover_relevance": pd.NA,
                "uncertainty": float("nan"),
                "evidence_types": [pd.NA, float("nan"), "forêt"],
            }
        ),
        article_type=pd.Series({"candidate_article_types": [pd.NA, "zone_urbaine"]}),
        spatial=pd.Series(
            {
                "point_label_share_250m": float("nan"),
                "point_label_share_500m": pd.NA,
                "dominant_matches_point_label_250m": pd.NA,
            }
        ),
        quality=pd.Series(
            {
                "quality_score": float("nan"),
                "quality_bin": pd.NA,
                "recommended_use": pd.NA,
            }
        ),
    )

    card = record["evidence_card"]

    assert "<NA>" not in card
    assert re.search(r"\bnan\b", card, flags=re.IGNORECASE) is None
    assert record["landcover_relevance"] is None
    assert record["uncertainty"] is None
    assert record["evidence_types"] == ["forêt"]
    assert record["candidate_article_types"] == ["zone_urbaine"]
    assert record["point_label_share_250m"] is None
    assert record["point_label_share_500m"] is None
    assert record["dominant_matches_point_label_250m"] is None
    assert record["quality_score"] is None
    assert record["quality_bin"] is None
    assert record["recommended_use"] is None

    assert json.dumps(record, allow_nan=False)
