"""Deterministic evidence-highlight text artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from georeset.analysis.list_normalization import normalize_string_list
from georeset.text.labels import EVIDENCE_TYPE_LABELS, format_list
from georeset.text.record_access import json_scalar, mapping_get
from georeset.text.title_scrubbing import remove_title_variants

EVIDENCE_HIGHLIGHTS_VERSION = 1


def _clean_sentences(raw_sentences: object, title: str) -> list[str]:
    sentences = normalize_string_list(raw_sentences)
    return [
        cleaned for sentence in sentences if (cleaned := remove_title_variants(sentence, title))
    ]


def _label_evidence_types(evidence_types: list[str]) -> list[str]:
    return [EVIDENCE_TYPE_LABELS.get(item, item.replace("_", " ")) for item in evidence_types]


def _build_highlight_text(
    *,
    title: str,
    evidence: Mapping[str, Any] | pd.Series | None,
) -> tuple[str, int]:
    evidence_types = normalize_string_list(mapping_get(evidence, "evidence_types", []))
    sentences = _clean_sentences(mapping_get(evidence, "evidence_sentences_no_place", []), title)
    sentence_lines = (
        [f"- {sentence}" for sentence in sentences]
        if sentences
        else ["- Aucune phrase d'indice explicite n'est disponible."]
    )
    relevance = json_scalar(mapping_get(evidence, "landcover_relevance")) or "inconnue"
    uncertainty = json_scalar(mapping_get(evidence, "uncertainty")) or "inconnue"
    text = "\n".join(
        [
            "Indices d'occupation du sol extraits, sans nom de lieu.",
            "",
            f"Pertinence des indices: {relevance}",
            f"Incertitude des indices: {uncertainty}",
            f"Types d'indices: {format_list(_label_evidence_types(evidence_types))}",
            "",
            "Phrases d'indices:",
            *sentence_lines,
        ]
    )
    return text, len(sentences)


def build_evidence_highlight_record(
    *,
    pageid: str,
    article: Mapping[str, Any],
    evidence: Mapping[str, Any] | pd.Series | None,
) -> dict[str, Any]:
    """Build one deterministic highlighted-content record."""
    title = str(article.get("title") or "").strip()
    raw_content = str(article.get("content") or "").strip()
    highlights, sentence_count = _build_highlight_text(title=title, evidence=evidence)
    content_with_highlights = f"{highlights}\n\nTexte complet de l'article:\n{raw_content}"
    evidence_types = normalize_string_list(mapping_get(evidence, "evidence_types", []))
    record = {
        "pageid": str(pageid),
        "title": title,
        "url": article.get("url", ""),
        "evidence_highlights": highlights,
        "content_with_evidence_highlights": content_with_highlights,
        "landcover_relevance": json_scalar(mapping_get(evidence, "landcover_relevance")),
        "uncertainty": json_scalar(mapping_get(evidence, "uncertainty")),
        "evidence_types": evidence_types,
        "evidence_sentence_count": sentence_count,
        "evidence_highlights_char_count": len(highlights),
        "content_with_evidence_highlights_char_count": len(content_with_highlights),
        "metadata": {
            "source": "deterministic_evidence_highlights",
            "version": EVIDENCE_HIGHLIGHTS_VERSION,
            "uses_raw_content": True,
            "inputs": ["article_contents", "article_landuse_evidence_summaries"],
        },
    }
    return record
