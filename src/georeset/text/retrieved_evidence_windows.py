"""Deterministic raw-sentence retrieval artifacts for classification experiments."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from georeset.analysis.list_normalization import normalize_string_list
from georeset.text.record_access import json_scalar, mapping_get
from georeset.text.title_scrubbing import remove_title_variants

RETRIEVED_EVIDENCE_WINDOWS_VERSION = 1
EVIDENCE_TYPE_KEYWORDS = {
    "agriculture": ("agricole", "agriculture", "culture", "cultures", "terres agricoles"),
    "bare_ground": ("roche", "éboulis", "sable", "sol nu", "improductif"),
    "forest": ("forêt", "forêts", "forestier", "bois", "boisé", "boisée", "boisements"),
    "habitat_or_ecology": ("habitat", "écologie", "écologique", "réserve naturelle", "protégé"),
    "orchard": ("verger", "vergers"),
    "pasture": ("prairie", "prairies", "pâturage", "pâturages", "herbage"),
    "relief_or_geology": ("altitude", "montagne", "relief", "géologie", "versant", "vallée"),
    "shrubland": ("lande", "landes", "broussailles", "arbustes", "pelouse"),
    "urban_or_artificial": ("bâtiments", "route", "routes", "urbanisé", "aménagé"),
    "vineyard": ("vigne", "vignes", "vignoble", "viticole"),
    "water": ("rivière", "rivières", "lac", "lacs", "eau", "cours d eau", "étang"),
    "wetland": ("zone humide", "zones humides", "tourbière", "marais", "humide"),
}


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()
    ]


def _normalized_match_text(text: str) -> str:
    return re.sub(r"\W+", " ", text, flags=re.UNICODE).casefold().strip()


def _matched_sentence_indices(sentences: list[str], evidence_sentences: list[str]) -> list[int]:
    normalized_evidence = [_normalized_match_text(sentence) for sentence in evidence_sentences]
    matches: list[int] = []
    for index, sentence in enumerate(sentences):
        normalized_sentence = _normalized_match_text(sentence)
        if any(
            evidence and (evidence in normalized_sentence or normalized_sentence in evidence)
            for evidence in normalized_evidence
        ):
            matches.append(index)
    return matches


def _keyword_sentence_indices(sentences: list[str], evidence_types: list[str]) -> list[int]:
    keywords = [
        _normalized_match_text(keyword)
        for evidence_type in evidence_types
        for keyword in EVIDENCE_TYPE_KEYWORDS.get(evidence_type, ())
    ]
    if not keywords:
        return []
    return [
        index
        for index, sentence in enumerate(sentences)
        if any(keyword in _normalized_match_text(sentence) for keyword in keywords)
    ]


def _window_indices(matches: list[int], sentence_count: int, radius: int) -> list[int]:
    indices = {
        index
        for match in matches
        for index in range(max(0, match - radius), min(sentence_count, match + radius + 1))
    }
    return sorted(indices)


def _stable_rng(seed: int, pageid: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{pageid}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _random_indices(sentence_count: int, sample_size: int, *, seed: int, pageid: str) -> list[int]:
    if sentence_count == 0 or sample_size <= 0:
        return []
    rng = _stable_rng(seed, pageid)
    return sorted(rng.sample(range(sentence_count), k=min(sentence_count, sample_size)))


def _format_sentences(title: str, sentences: list[str]) -> str:
    body = [f"- {sentence}" for sentence in sentences] or ["- Aucune phrase n'est disponible."]
    return "\n".join([title, "", *body])


def build_retrieved_evidence_window_record(
    *,
    pageid: str,
    article: Mapping[str, Any],
    evidence: Mapping[str, Any] | pd.Series | None,
    seed: int,
    context_radius: int = 1,
) -> dict[str, Any]:
    """Build deterministic retrieved/raw/random sentence-window text sources."""
    title = str(article.get("title") or "").strip()
    content = str(article.get("content") or "").strip()
    sentences = _split_sentences(content)
    evidence_sentences = normalize_string_list(
        mapping_get(evidence, "evidence_sentences_no_place", [])
    )
    matches = _matched_sentence_indices(sentences, evidence_sentences)
    if not matches:
        matches = _keyword_sentence_indices(
            sentences, normalize_string_list(mapping_get(evidence, "evidence_types", []))
        )
    retrieved_indices = _window_indices(matches, len(sentences), context_radius)
    if not retrieved_indices and sentences:
        retrieved_indices = [0]
    sentence_only_indices = matches or retrieved_indices[:1]
    random_indices = _random_indices(
        len(sentences), len(retrieved_indices), seed=seed, pageid=str(pageid)
    )
    retrieved_sentences = [sentences[index] for index in retrieved_indices]
    sentence_only = [sentences[index] for index in sentence_only_indices]
    random_sentences = [sentences[index] for index in random_indices]
    no_place_sentences = [
        remove_title_variants(sentence, title) for sentence in retrieved_sentences
    ]

    retrieved_text = _format_sentences(
        "Phrases brutes extraites de l'article.", retrieved_sentences
    )
    sentence_only_text = _format_sentences(
        "Phrases d'indice extraites de l'article.", sentence_only
    )
    random_text = _format_sentences(
        "Phrases brutes aléatoires extraites du même article.", random_sentences
    )
    no_place_text = _format_sentences(
        "Phrases brutes extraites de l'article, sans nom de lieu.",
        [sentence for sentence in no_place_sentences if sentence],
    )
    return {
        "pageid": str(pageid),
        "title": title,
        "url": article.get("url", ""),
        "retrieved_evidence_windows": retrieved_text,
        "retrieved_evidence_sentences_only": sentence_only_text,
        "random_sentence_windows": random_text,
        "retrieved_evidence_windows_no_place": no_place_text,
        "landcover_relevance": json_scalar(mapping_get(evidence, "landcover_relevance")),
        "uncertainty": json_scalar(mapping_get(evidence, "uncertainty")),
        "evidence_types": normalize_string_list(mapping_get(evidence, "evidence_types", [])),
        "article_sentence_count": len(sentences),
        "matched_evidence_sentence_count": len(matches),
        "retrieved_sentence_count": len(retrieved_sentences),
        "random_sentence_count": len(random_sentences),
        "retrieved_evidence_windows_char_count": len(retrieved_text),
        "retrieved_evidence_sentences_only_char_count": len(sentence_only_text),
        "random_sentence_windows_char_count": len(random_text),
        "retrieved_evidence_windows_no_place_char_count": len(no_place_text),
        "metadata": {
            "source": "deterministic_retrieved_evidence_windows",
            "version": RETRIEVED_EVIDENCE_WINDOWS_VERSION,
            "uses_raw_content": True,
            "seed": seed,
            "context_radius": context_radius,
            "inputs": ["article_contents", "article_landuse_evidence_summaries"],
        },
    }
