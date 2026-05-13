"""Tests for deterministic Wikipedia article-type assignment."""

from georeset.analysis.article_type_classifier import (
    assign_article_types,
    normalize_text,
)


def test_multiple_matches_preserve_candidates_and_respect_precedence() -> None:
    categories = [
        "Catégorie:Rivière du Verdon",
        "Catégorie:Col des Écrins",
        "Catégorie:Vignoble du Sancerre",
        "Catégorie:Gare de Mulhouse",
    ]

    assignment = assign_article_types(categories)

    assert assignment.primary_article_type == "water_feature"
    assert assignment.candidate_article_types == [
        "water_feature",
        "natural_landscape",
        "agriculture_or_vineyard",
        "transport_infrastructure",
    ]
    assert any("col" in category for category in assignment.matched_categories)
    assert "vignoble" in assignment.matched_rules
    assert "gare" in assignment.matched_rules


def test_unknown_or_missing_categories_become_other_or_unclear() -> None:
    assert assign_article_types(None).primary_article_type == "other_or_unclear"
    assert assign_article_types([]).candidate_article_types == ["other_or_unclear"]


def test_candidate_matching_is_case_and_accent_insensitive() -> None:
    assignment = assign_article_types(["Catégorie:PRéfecture d'une Commune", "Catégorie:Vallée de la Loire"])

    assert assignment.primary_article_type == "natural_landscape"
    assert "natural_landscape" in assignment.candidate_article_types
    assert "commune" in assignment.matched_rules
    assert any("vallee" in category for category in assignment.matched_categories)


def test_matched_categories_are_preserved_and_normalized() -> None:
    assignment = assign_article_types(["Catégorie:Rivière de la Loire", "  Catégorie:Vignoble des Côtes-du-Rhône "])

    assert assignment.matched_categories == ["riviere de la loire", "vignoble des cotes-du-rhone"]
    assert assignment.matched_rules == ["rivière", "vignoble"]


def test_normalize_text_removes_prefix_and_accents() -> None:
    assert normalize_text("  Catégorie:Église-Sainte  ") == "eglise-sainte"
    assert normalize_text("RÂPEUR") == "rapeur"
    assert normalize_text("  Catégorie: vallée ") == "vallee"


def test_prairie_generically_maps_to_agriculture_or_vineyard() -> None:
    assignment = assign_article_types(["Catégorie:Prairie"])

    assert assignment.primary_article_type == "agriculture_or_vineyard"
    assert assignment.candidate_article_types == ["agriculture_or_vineyard"]


def test_prairie_naturelle_maps_to_natural_landscape() -> None:
    assignment = assign_article_types(["Catégorie:Prairie naturelle"])

    assert assignment.primary_article_type == "natural_landscape"
    assert "natural_landscape" in assignment.candidate_article_types


def test_prairie_agricole_not_match_landscape_col_rule() -> None:
    assignment = assign_article_types(["Catégorie:Prairie agricole"])

    assert assignment.primary_article_type == "agriculture_or_vineyard"
    assert assignment.candidate_article_types == ["agriculture_or_vineyard"]


def test_school_category_does_not_match_col_substring() -> None:
    assignment = assign_article_types(["Catégorie:École communale"])

    assert "natural_landscape" not in assignment.candidate_article_types


def test_col_category_includes_natural_landscape() -> None:
    assignment = assign_article_types(["Catégorie:Col des Vosges"])

    assert "natural_landscape" in assignment.candidate_article_types
    assert assignment.primary_article_type == "natural_landscape"


def test_canal_de_navigation_includes_water_and_transport() -> None:
    assignment = assign_article_types(["Catégorie:Canal de navigation en France"])

    assert assignment.primary_article_type == "water_feature"
    assert "water_feature" in assignment.candidate_article_types
    assert "transport_infrastructure" in assignment.candidate_article_types


def test_cours_d_eau_punctuation_variant() -> None:
    assignment = assign_article_types(["Catégorie:Cours d'eau de la Garonne"])

    assert "water_feature" in assignment.candidate_article_types
