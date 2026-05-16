"""Shared display labels for deterministic text artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from georeset.text.record_access import is_missing

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
    "forest": "forêt",
    "agriculture": "agriculture",
    "vineyard": "vignoble",
    "pasture": "prairie ou pâturage",
    "water": "eau",
    "wetland": "zone humide",
    "shrubland": "végétation arbustive ou herbacée",
    "bare_ground": "sol nu ou rocheux",
    "urban_or_artificial": "urbain ou artificiel",
    "relief_or_geology": "relief ou géologie",
    "habitat_or_ecology": "habitat ou écologie",
}


def enum_text(
    value: object,
    label_map: Mapping[str, str],
    *,
    default: str = "inconnue",
) -> str:
    """Render an enum-like value using a label map."""
    if is_missing(value):
        return default
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in label_map:
        return label_map[normalized]
    return normalized.replace("_", " ")


def format_list(values: list[str]) -> str:
    """Render text values as a comma-separated phrase."""
    return ", ".join(values) if values else "aucun"
