"""Text-source policies for article classification experiments."""

import random

SHUFFLED_TEXT_SOURCES = {
    "summary_shuffled": "summary",
    "summary_no_place_shuffled": "summary_no_place",
    "content_shuffled": "content",
    "landuse_evidence_summary_shuffled": "landuse_evidence_summary",
    "evidence_card_shuffled": "evidence_card",
    "content_with_evidence_card_shuffled": "content_with_evidence_card",
    "content_with_evidence_highlights_shuffled": "content_with_evidence_highlights",
}
TEXT_SOURCE_CHOICES = [
    "summary",
    "summary_no_place",
    "landuse_evidence_summary",
    "evidence_card",
    "content_with_evidence_card",
    "content_with_evidence_highlights",
    "content",
    "summary_shuffled",
    "summary_no_place_shuffled",
    "landuse_evidence_summary_shuffled",
    "evidence_card_shuffled",
    "content_with_evidence_card_shuffled",
    "content_with_evidence_highlights_shuffled",
    "content_shuffled",
]


def base_text_source(text_source: str) -> str:
    return SHUFFLED_TEXT_SOURCES.get(text_source, text_source)


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
