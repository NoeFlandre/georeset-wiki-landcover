"""Deterministic Wikipedia article-type classifier from category labels."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

ArticleType = str

ARTICLE_TYPE_PREFERENCE: list[ArticleType] = [
    "water_feature",
    "natural_landscape",
    "agriculture_or_vineyard",
    "built_or_cultural_site",
    "transport_infrastructure",
    "settlement_or_administrative",
    "person_or_event",
    "other_or_unclear",
]

ARTICLE_TYPE_CANDIDATES: set[str] = {
    "water_feature",
    "natural_landscape",
    "agriculture_or_vineyard",
    "built_or_cultural_site",
    "transport_infrastructure",
    "settlement_or_administrative",
    "person_or_event",
    "other_or_unclear",
}

# Ordered mapping preserves deterministic first-match behavior.
ARTICLE_TYPE_PATTERNS: list[tuple[ArticleType, list[str]]] = [
    ("water_feature", ["rivière", "fleuve", "ruisseau", "lac", "étang", "etang", "canal", "cours d'eau", "cascade", "source", "hydrologie", "hydrographie"]),
    (
        "natural_landscape",
        [
            "montagne",
            "massif",
            "vallée",
            "vallee",
            "forêt",
            "foret",
            "bois",
            "réserve naturelle",
            "reserve naturelle",
            "parc naturel",
            "zone naturelle",
            "sommet",
            "col",
            "paysage",
            "géologie",
            "geologie",
            "falaise",
            "grotte",
            "prairie naturelle",
        ],
    ),
    (
        "agriculture_or_vineyard",
        [
            "vignoble",
            "vin",
            "viticulture",
            "agriculture",
            "agricole",
            "culture",
            "céréale",
            "cereale",
            "prairie",
            "paturage",
            "verger",
            "exploitation agricole",
        ],
    ),
    (
        "built_or_cultural_site",
        [
            "château",
            "chateau",
            "église",
            "eglise",
            "abbaye",
            "monument",
            "musée",
            "musee",
            "bâtiment",
            "batiment",
            "patrimoine",
            "site archeologique",
            "architecture",
            "édifice",
            "edifice",
            "chapelle",
        ],
    ),
    (
        "transport_infrastructure",
        [
            "gare",
            "route",
            "autoroute",
            "canal de navigation",
            "voie ferrée",
            "voie ferree",
            "pont",
            "tunnel",
            "transport",
        ],
    ),
    (
        "settlement_or_administrative",
        [
            "commune",
            "village",
            "localité",
            "localite",
            "département",
            "departement",
            "canton",
            "arrondissement",
            "intercommunalité",
            "intercommunalite",
            "ancienne commune",
            "quartier",
            "lieu-dit",
        ],
    ),
    (
        "person_or_event",
        [
            "naissance",
            "décès",
            "deces",
            "personnalité",
            "personnalite",
            "personne",
            "bataille",
            "événement",
            "evenement",
            "biographie",
            "homme politique",
            "ecrivain",
            "écrivain",
            "artiste",
        ],
    ),
]


def normalize_text(text: str) -> str:
    """Normalize case, wiki category prefix, whitespace and accents.

    The normalization is intentionally simple so it can be used for both matching and
    deterministic output snapshots.
    """

    value = text.strip().lower()
    for prefix in ("catégorie:", "categorie:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    value = " ".join(value.replace("_", " ").split())
    decomposed = unicodedata.normalize("NFD", value)
    no_accents = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return no_accents


@dataclass(frozen=True)
class ArticleTypeAssignment:
    """Structured article-type assignment for one wiki page."""

    pageid: str
    title: str
    primary_article_type: str
    candidate_article_types: list[str]
    matched_categories: list[str]
    matched_rules: list[str]
    all_categories_count: int
    has_categories: bool


def _extract_categories(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _normalise_input_categories(raw_categories: Any) -> list[str]:
    output: list[str] = []
    for item in _extract_categories(raw_categories):
        normalized = normalize_text(item)
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _normalize_for_match(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _pattern_matches(text: str, raw_pattern: str) -> bool:
    pattern = _normalize_for_match(raw_pattern)
    if not pattern:
        return False

    if " " in pattern:
        tokens = pattern.split()
        token_patterns = [
            rf"{re.escape(token)}s?" if len(token) <= 4 and token.isalpha() else re.escape(token)
            for token in tokens
        ]
        token_sep = r"\s+"
        regex = re.compile(rf"(?<![a-z0-9])(?:{token_sep.join(token_patterns)})(?![a-z0-9])")
    else:
        token_pattern = rf"{re.escape(pattern)}s?" if len(pattern) <= 4 and pattern.isalpha() else re.escape(pattern)
        regex = re.compile(rf"(?<![a-z0-9])(?:{token_pattern})(?![a-z0-9])")

    normalized_text = _normalize_for_match(text)
    return bool(regex.search(normalized_text))


def _candidate_types(normalized_categories: list[str]) -> tuple[list[str], list[str], list[str]]:
    candidate_types: list[str] = []
    matched_categories: list[str] = []
    matched_rules: list[str] = []
    matched_rule_keys: set[str] = set()

    for category in normalized_categories:
        for article_type, patterns in ARTICLE_TYPE_PATTERNS:
            for pattern in patterns:
                if _pattern_matches(category, pattern):
                    normalized_pattern = normalize_text(pattern)
                    if not normalized_pattern:
                        continue
                    if article_type not in candidate_types:
                        candidate_types.append(article_type)
                    if category not in matched_categories:
                        matched_categories.append(category)
                    if normalized_pattern not in matched_rule_keys:
                        matched_rules.append(pattern)
                        matched_rule_keys.add(normalized_pattern)

    ordered_candidates = [
        article_type for article_type in ARTICLE_TYPE_PREFERENCE[:-1] if article_type in candidate_types
    ]
    if not ordered_candidates:
        return ["other_or_unclear"], [], []
    return ordered_candidates, matched_categories, matched_rules


def assign_article_types(
    raw_categories: Any,
    *,
    pageid: str | int = "",
    title: str = "",
) -> ArticleTypeAssignment:
    """Assign primary and candidate article types from category metadata."""

    normalized_categories = _normalise_input_categories(raw_categories)
    candidate_types, matched_categories, matched_rules = _candidate_types(normalized_categories)

    primary_article_type = candidate_types[0] if candidate_types else "other_or_unclear"
    all_categories_count = len(normalized_categories)
    return ArticleTypeAssignment(
        pageid=str(pageid),
        title=str(title),
        primary_article_type=primary_article_type,
        candidate_article_types=candidate_types,
        matched_categories=matched_categories,
        matched_rules=matched_rules,
        all_categories_count=all_categories_count,
        has_categories=all_categories_count > 0,
    )


def assign_from_metadata_record(record: dict[str, Any]) -> ArticleTypeAssignment:
    """Convenience wrapper for a metadata payload containing page fields."""

    pageid = str(record.get("pageid", ""))
    title = str(record.get("title", ""))
    return assign_article_types(record.get("categories"), pageid=pageid, title=title)
