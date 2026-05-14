"""Deterministic evidence-highlight text artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from georeset.analysis.list_normalization import normalize_string_list
from georeset.text.evidence_cards import EVIDENCE_TYPE_LABELS

EVIDENCE_HIGHLIGHTS_VERSION = 1


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return False
    return bool(pd.isna(value))


def _json_scalar(value: object) -> object | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    return value


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
    return re.compile(r"[\W_]+".join(re.escape(token) for token in tokens), flags=re.IGNORECASE)


def _remove_title_variants(text: str, title: str) -> str:
    title = title.strip()
    if title:
        text = re.sub(re.escape(title), "ce lieu", text, flags=re.IGNORECASE)
    pattern = _title_pattern(title)
    if pattern is not None:
        text = pattern.sub("ce lieu", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_sentences(raw_sentences: object, title: str) -> list[str]:
    sentences = normalize_string_list(raw_sentences)
    return [
        cleaned
        for sentence in sentences
        if (cleaned := _remove_title_variants(sentence, title))
    ]


def _label_evidence_types(evidence_types: list[str]) -> list[str]:
    return [EVIDENCE_TYPE_LABELS.get(item, item.replace("_", " ")) for item in evidence_types]


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "aucun"


def _build_highlight_text(
    *,
    title: str,
    evidence: Mapping[str, Any] | pd.Series | None,
) -> tuple[str, int]:
    evidence_types = normalize_string_list(_mapping_get(evidence, "evidence_types", []))
    sentences = _clean_sentences(_mapping_get(evidence, "evidence_sentences_no_place", []), title)
    sentence_lines = (
        [f"- {sentence}" for sentence in sentences]
        if sentences
        else ["- Aucune phrase d'indice explicite n'est disponible."]
    )
    relevance = _json_scalar(_mapping_get(evidence, "landcover_relevance")) or "inconnue"
    uncertainty = _json_scalar(_mapping_get(evidence, "uncertainty")) or "inconnue"
    text = "\n".join(
        [
            "Indices d'occupation du sol extraits, sans nom de lieu.",
            "",
            f"Pertinence des indices: {relevance}",
            f"Incertitude des indices: {uncertainty}",
            f"Types d'indices: {_format_list(_label_evidence_types(evidence_types))}",
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
    evidence_types = normalize_string_list(_mapping_get(evidence, "evidence_types", []))
    record = {
        "pageid": str(pageid),
        "title": title,
        "url": article.get("url", ""),
        "evidence_highlights": highlights,
        "content_with_evidence_highlights": content_with_highlights,
        "landcover_relevance": _json_scalar(_mapping_get(evidence, "landcover_relevance")),
        "uncertainty": _json_scalar(_mapping_get(evidence, "uncertainty")),
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
