"""Deterministic evidence-card text artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

EVIDENCE_CARD_VERSION = 1


ARTICLE_TYPE_LABELS = {
    "other_or_unclear": "autre / non précisé",
    "water_feature": "caractéristique hydrographique",
    "natural_landscape": "paysage naturel",
    "agriculture_or_vineyard": "agriculture ou vignoble",
    "built_or_cultural_site": "site bâti ou culturel",
    "transport_infrastructure": "infrastructure de transport",
    "settlement_or_administrative": "zone urbaine / administrative",
    "person_or_event": "personne ou événement",
}
QUALITY_BIN_LABELS = {
    "quality_low": "qualité faible",
    "quality_medium": "qualité moyenne",
    "quality_high": "qualité élevée",
    "quality_very_high": "qualité très élevée",
}
RECOMMENDED_USE_LABELS = {
    "use_for_training": "utilisable pour entraînement",
    "use_for_evaluation_only": "réservé à l'évaluation",
    "inspect_manually": "à inspecter manuellement",
    "exclude": "à exclure",
}
LANDCOVER_RELEVANCE_LABELS = {
    "none": "aucune",
    "low": "faible",
    "medium": "moyenne",
    "high": "élevée",
}
UNCERTAINTY_LABELS = {
    "low": "faible",
    "medium": "moyenne",
    "high": "élevée",
}
EVIDENCE_TYPE_LABELS = {
    "water": "eau",
    "forest": "forêt",
    "agriculture": "agriculture",
    "urban_or_artificial": "urbain ou artificiel",
    "vineyard": "vignoble",
}


def _enum_text(
    value: object,
    label_map: Mapping[str, str],
    *,
    default: str = "inconnue",
) -> str:
    if _is_missing(value):
        return default
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in label_map:
        return label_map[normalized]
    return normalized.replace("_", " ")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return False
    return bool(pd.isna(value))


def _text(value: object, default: str = "inconnue") -> str:
    if _is_missing(value):
        return default
    return str(value).strip()


def _number_text(value: object) -> str:
    if _is_missing(value):
        return "inconnue"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "inconnue"
    return f"{float(numeric):.3f}".rstrip("0").rstrip(".")


def _bool_text(value: object) -> str:
    if isinstance(value, bool):
        return "oui" if value else "non"
    if _is_missing(value):
        return "inconnue"
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "oui"}:
        return "oui"
    if normalized in {"false", "0", "no", "n", "non"}:
        return "non"
    return "inconnue"


def _list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if not _is_missing(item) and str(item).strip()]
    if isinstance(value, tuple):
        return _list(list(value))
    if _is_missing(value):
        return []
    return [str(value).strip()]


def _json_scalar(value: Any, *, default: Any = None) -> Any:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):
        converted = value.item()
        return _json_scalar(converted, default=default)
    return value


def _json_list(value: object) -> list[Any]:
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, tuple):
        raw_values = list(value)
    elif _is_missing(value):
        return []
    else:
        raw_values = [value]
    values: list[Any] = []
    for item in raw_values:
        cleaned = _json_scalar(item)
        if cleaned is not None:
            values.append(cleaned)
    return values


def _mapping_get(mapping: Mapping[str, Any] | pd.Series | None, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, pd.Series):
        return mapping.get(key, default)
    return mapping.get(key, default)


def _title_pattern(title: str) -> re.Pattern[str] | None:
    tokens = [token for token in re.split(r"[\W_]+", title, flags=re.UNICODE) if token]
    if not tokens:
        return None
    separator = r"[\W_]+"
    return re.compile(separator.join(re.escape(token) for token in tokens), flags=re.IGNORECASE)


def _exact_title_pattern(title: str) -> re.Pattern[str] | None:
    title = title.strip()
    if not title:
        return None
    return re.compile(re.escape(title), flags=re.IGNORECASE)


def _remove_title_variants(text: str, title: str) -> str:
    exact_pattern = _exact_title_pattern(title)
    if exact_pattern is not None:
        text = exact_pattern.sub("ce lieu", text)
    pattern = _title_pattern(title)
    if pattern is None:
        return text
    cleaned = pattern.sub("ce lieu", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _evidence_sentences(evidence: Mapping[str, Any] | pd.Series | None, title: str) -> list[str]:
    raw = _mapping_get(evidence, "evidence_sentences_no_place", [])
    sentences = [_remove_title_variants(sentence, title) for sentence in _list(raw)]
    return [sentence for sentence in sentences if sentence]


def _summary(evidence: Mapping[str, Any] | pd.Series | None, title: str) -> str:
    summary = _text(
        _mapping_get(evidence, "landuse_evidence_summary"),
        default="Aucun résumé d'indices n'est disponible.",
    )
    return _remove_title_variants(summary, title)


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "aucun"


def _build_card_text(
    *,
    title: str,
    evidence: Mapping[str, Any] | pd.Series | None,
    article_type: Mapping[str, Any] | pd.Series | None,
    spatial: Mapping[str, Any] | pd.Series | None,
    quality: Mapping[str, Any] | pd.Series | None,
) -> tuple[str, int]:
    evidence_types = _list(_mapping_get(evidence, "evidence_types", []))
    candidate_article_types = _list(_mapping_get(article_type, "candidate_article_types", []))
    evidence_type_labels = [_enum_text(item, EVIDENCE_TYPE_LABELS) for item in evidence_types]
    candidate_article_type_labels = [
        _enum_text(item, ARTICLE_TYPE_LABELS) for item in candidate_article_types
    ]
    sentences = _evidence_sentences(evidence, title)
    sentence_lines = (
        [f"- {sentence}" for sentence in sentences]
        if sentences
        else ["- Aucun indice factuel explicite n'a été extrait."]
    )
    card = "\n".join(
        [
            "Fiche d'indices d'occupation du sol, sans nom de lieu.",
            "",
            f"Pertinence: {_enum_text(_mapping_get(evidence, 'landcover_relevance'), LANDCOVER_RELEVANCE_LABELS)}",
            f"Incertitude: {_enum_text(_mapping_get(evidence, 'uncertainty'), UNCERTAINTY_LABELS)}",
            f"Types d'indices: {_format_list(evidence_type_labels)}",
            f"Type d'article: {_enum_text(_mapping_get(article_type, 'primary_article_type'), ARTICLE_TYPE_LABELS)}",
            f"Types d'article candidats: {_format_list(candidate_article_type_labels)}",
            f"Score de qualité: {_text(_mapping_get(quality, 'quality_score'))}",
            f"Catégorie de qualité: {_enum_text(_mapping_get(quality, 'quality_bin'), QUALITY_BIN_LABELS)}",
            f"Catégorie d'usage recommandée: {_enum_text(_mapping_get(quality, 'recommended_use'), RECOMMENDED_USE_LABELS)}",
            "",
            "Confiance spatiale CORINE:",
            f"- part du label ponctuel à 250 m: {_number_text(_mapping_get(spatial, 'point_label_share_250m'))}",
            f"- part du label ponctuel à 500 m: {_number_text(_mapping_get(spatial, 'point_label_share_500m'))}",
            "- le label dominant à 250 m correspond au label ponctuel: "
            f"{_bool_text(_mapping_get(spatial, 'dominant_matches_point_label_250m'))}",
            "",
            "Indices factuels:",
            *sentence_lines,
            "",
            "Résumé d'indices:",
            _summary(evidence, title),
        ]
    )
    return card, len(sentences)


def build_evidence_card_record(
    *,
    pageid: str,
    article: Mapping[str, Any],
    evidence: Mapping[str, Any] | pd.Series | None,
    article_type: Mapping[str, Any] | pd.Series | None,
    spatial: Mapping[str, Any] | pd.Series | None,
    quality: Mapping[str, Any] | pd.Series | None,
) -> dict[str, Any]:
    """Build one deterministic evidence-card record."""
    title = _text(article.get("title"), default="")
    raw_content = _text(article.get("content"), default="")
    evidence_card, sentence_count = _build_card_text(
        title=title,
        evidence=evidence,
        article_type=article_type,
        spatial=spatial,
        quality=quality,
    )
    content_with_evidence_card = (
        f"{evidence_card}\n\nTexte complet de l'article:\n{raw_content}"
    )
    evidence_types = _json_list(_mapping_get(evidence, "evidence_types", []))
    candidate_article_types = _json_list(_mapping_get(article_type, "candidate_article_types", []))
    record = {
        "pageid": str(pageid),
        "title": title,
        "url": article.get("url", ""),
        "evidence_card": evidence_card,
        "content_with_evidence_card": content_with_evidence_card,
        "landcover_relevance": _json_scalar(_mapping_get(evidence, "landcover_relevance")),
        "uncertainty": _json_scalar(_mapping_get(evidence, "uncertainty")),
        "evidence_types": evidence_types,
        "evidence_sentences_count": sentence_count,
        "landuse_evidence_summary_char_count": _json_scalar(
            _mapping_get(evidence, "landuse_evidence_summary_char_count", 0)
        ),
        "primary_article_type": _json_scalar(
            _mapping_get(article_type, "primary_article_type", "other_or_unclear"),
            default="other_or_unclear",
        ),
        "candidate_article_types": candidate_article_types,
        "point_label_share_250m": _json_scalar(_mapping_get(spatial, "point_label_share_250m")),
        "point_label_share_500m": _json_scalar(_mapping_get(spatial, "point_label_share_500m")),
        "dominant_matches_point_label_250m": _json_scalar(
            _mapping_get(spatial, "dominant_matches_point_label_250m")
        ),
        "quality_score": _json_scalar(_mapping_get(quality, "quality_score")),
        "quality_bin": _json_scalar(_mapping_get(quality, "quality_bin")),
        "recommended_use": _json_scalar(_mapping_get(quality, "recommended_use")),
        "evidence_card_char_count": len(evidence_card),
        "content_with_evidence_card_char_count": len(content_with_evidence_card),
        "evidence_sentence_count": sentence_count,
        "metadata": {
            "source": "deterministic_evidence_card",
            "version": EVIDENCE_CARD_VERSION,
            "inputs": [
                "article_contents",
                "article_landuse_evidence_summaries",
                "article_type_assignments",
                "spatial_confidence",
                "quality_scores",
            ],
            "text_variants": {
                "evidence_card": {"uses_raw_content": False},
                "content_with_evidence_card": {"uses_raw_content": True},
            },
        },
    }
    return record
