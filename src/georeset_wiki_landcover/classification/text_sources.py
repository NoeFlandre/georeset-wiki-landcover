"""Text-source policies for article classification experiments."""

import random
from collections.abc import Collection

SHUFFLED_TEXT_SOURCES = {
    "summary_shuffled": "summary",
    "summary_no_place_shuffled": "summary_no_place",
    "content_shuffled": "content",
    "landuse_evidence_summary_shuffled": "landuse_evidence_summary",
    "evidence_card_shuffled": "evidence_card",
    "content_with_evidence_card_shuffled": "content_with_evidence_card",
    "content_with_evidence_highlights_shuffled": "content_with_evidence_highlights",
    "retrieved_evidence_windows_shuffled": "retrieved_evidence_windows",
}
TEXT_SOURCE_CHOICES = [
    "summary",
    "summary_no_place",
    "landuse_evidence_summary",
    "evidence_card",
    "content_with_evidence_card",
    "content_with_evidence_highlights",
    "retrieved_evidence_windows",
    "retrieved_evidence_sentences_only",
    "random_sentence_windows",
    "retrieved_evidence_windows_no_place",
    "content",
    "summary_shuffled",
    "summary_no_place_shuffled",
    "landuse_evidence_summary_shuffled",
    "evidence_card_shuffled",
    "content_with_evidence_card_shuffled",
    "content_with_evidence_highlights_shuffled",
    "retrieved_evidence_windows_shuffled",
    "content_shuffled",
]
TEXT_SOURCE_ORDER = {text_source: index for index, text_source in enumerate(TEXT_SOURCE_CHOICES)}


def base_text_source(text_source: str) -> str:
    return SHUFFLED_TEXT_SOURCES.get(text_source, text_source)


def text_source_sort_key(text_source: str) -> tuple[int, str]:
    """Return the stable experiment ordering key for a text source."""
    return (TEXT_SOURCE_ORDER.get(text_source, len(TEXT_SOURCE_ORDER)), text_source)


def shuffled_text_source_pairs(text_sources: Collection[str] | None = None) -> dict[str, str]:
    """Return base-to-shuffled pairs in stable text-source order.

    If ``text_sources`` is provided, only pairs where both base and shuffled sources
    are present are returned. This keeps analysis scripts from silently reporting
    deltas for sources that were not part of a specific experiment.
    """
    available = set(text_sources) if text_sources is not None else None
    pairs = {
        base_source: shuffled_source
        for shuffled_source, base_source in SHUFFLED_TEXT_SOURCES.items()
        if available is None or (base_source in available and shuffled_source in available)
    }
    return dict(sorted(pairs.items(), key=lambda item: text_source_sort_key(item[0])))


def apply_shuffled_text_control(
    text_records: dict[str, str], eligible: list[str], seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Return text records where eligible article texts are deterministically reassigned."""
    if len(eligible) <= 1:
        return dict(text_records), {pageid: pageid for pageid in eligible}

    shuffled_ids = list(eligible)
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(shuffled_ids)
        if all(
            pageid != source_pageid
            for pageid, source_pageid in zip(eligible, shuffled_ids, strict=True)
        ):
            break
    else:
        shuffled_ids = eligible[1:] + eligible[:1]

    shuffled_records = dict(text_records)
    mapping = dict(zip(eligible, shuffled_ids, strict=True))
    for pageid, source_pageid in mapping.items():
        shuffled_records[pageid] = text_records[source_pageid]
    return shuffled_records, mapping


def shuffled_metadata(text_source: str, shuffled_from_pageid: str) -> dict[str, str]:
    return {
        "text_control": "shuffled",
        "base_text_source": SHUFFLED_TEXT_SOURCES[text_source],
        "shuffled_from_pageid": shuffled_from_pageid,
    }
