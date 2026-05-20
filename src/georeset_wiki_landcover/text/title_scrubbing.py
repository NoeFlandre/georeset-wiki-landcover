"""Shared title-removal helpers for text artifacts."""

from __future__ import annotations

import re


def _title_pattern(title: str) -> re.Pattern[str] | None:
    tokens = [token for token in re.split(r"[\W_]+", title, flags=re.UNICODE) if token]
    if not tokens:
        return None
    return re.compile(r"[\W_]+".join(re.escape(token) for token in tokens), flags=re.IGNORECASE)


def remove_title_variants(text: str, title: str) -> str:
    """Replace exact and separator-varied title mentions with a neutral phrase."""
    title = title.strip()
    if title:
        text = re.sub(re.escape(title), "ce lieu", text, flags=re.IGNORECASE)
    pattern = _title_pattern(title)
    if pattern is not None:
        text = pattern.sub("ce lieu", text)
    return re.sub(r"\s+", " ", text).strip()
