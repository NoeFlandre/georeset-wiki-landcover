"""Article metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable

from georeset_wiki_landcover.contracts import ArticleMeta


def index_articles_by_pageid(articles: Iterable[ArticleMeta]) -> dict[str, ArticleMeta]:
    """Return a first-wins index of articles keyed by normalized pageid."""
    indexed: dict[str, ArticleMeta] = {}
    for article in articles:
        pageid = article.get("pageid")
        if pageid is None:
            continue
        normalized = str(pageid)
        if normalized not in indexed:
            indexed[normalized] = article
    return indexed
