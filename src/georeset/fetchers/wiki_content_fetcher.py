import logging
import os
import time
from typing import cast

import requests

from georeset.config import DataPaths
from georeset.contracts import ArticleContent, ArticleMeta
from georeset.utils.json_io import read_json_file, write_json_atomic

logger = logging.getLogger(__name__)


class WikiContentFetcher:
    """Fetches Wikipedia article content using the extracts API."""

    def __init__(self) -> None:
        self.api_url = "https://fr.wikipedia.org/w/api.php"
        self.headers = {
            "User-Agent": "GeoResetPipeline/1.0 (https://geo-reset.sylvainlobry.com/; research@sylvainlobry.com) python-requests/2.33.1"
        }

    def get_articles_content(self, pageids: list[int]) -> dict[int, ArticleContent]:
        """
        Fetch content for multiple Wikipedia articles.

        Uses batch requests for speed, then falls back to individual requests
        for articles that returned empty content (max 1 retry per article).

        Args:
            pageids: List of Wikipedia page IDs.

        Returns:
            Dict mapping pageid -> {title, content, url}.
            Skips pages with negative IDs (not found) or empty content.
        """
        results: dict[int, ArticleContent] = {}

        # First: batch request for speed
        batch_results = self._batch_fetch(pageids)

        # Filter out empty content - only keep articles with actual text
        valid_batch = {pid: data for pid, data in batch_results.items() if data["content"].strip()}
        results.update(valid_batch)

        # Find pids that returned empty or missing content
        fetched_pids = set(results.keys())
        empty_pids = [pid for pid in pageids if pid not in fetched_pids]

        # Fall back to individual requests for empty ones (1 retry max)
        if empty_pids:
            individual_results = self._individual_fetch(empty_pids, max_retries=1)
            results.update(individual_results)

        return results

    def _batch_fetch(self, pageids: list[int]) -> dict[int, ArticleContent]:
        """Batch fetch multiple articles at once for speed."""
        results: dict[int, ArticleContent] = {}

        for i in range(0, len(pageids), 50):
            batch = pageids[i : i + 50]
            pageids_str = "|".join(str(pid) for pid in batch)

            params = {
                "action": "query",
                "prop": "extracts",
                "pageids": pageids_str,
                "explaintext": True,
                "exlimit": 1,
                "format": "json",
            }

            for attempt in range(3):
                try:
                    response = requests.get(
                        self.api_url, params=params, headers=self.headers, timeout=30
                    )
                    response.raise_for_status()
                    time.sleep(1.0)

                    data = response.json()
                    pages = data.get("query", {}).get("pages", {})

                    for pageid_str, page_data in pages.items():
                        pageid = int(pageid_str)
                        if pageid < 0:
                            continue

                        title = page_data.get("title", "")
                        content = page_data.get("extract", "")
                        url = f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}"

                        results[pageid] = {"title": title, "content": content, "url": url}

                    break

                except (requests.RequestException, ValueError):
                    if attempt < 2:
                        time.sleep(2**attempt)
                    continue

        return results

    def _individual_fetch(
        self, pageids: list[int], max_retries: int = 3
    ) -> dict[int, ArticleContent]:
        """Fetch articles one at a time for reliability."""
        results: dict[int, ArticleContent] = {}

        for pageid in pageids:
            params = {
                "action": "query",
                "prop": "extracts",
                "pageids": str(pageid),
                "explaintext": True,
                "exlimit": 1,
                "format": "json",
            }

            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        self.api_url, params=params, headers=self.headers, timeout=30
                    )
                    response.raise_for_status()
                    time.sleep(0.5)

                    data = response.json()
                    pages = data.get("query", {}).get("pages", {})

                    for pageid_str, page_data in pages.items():
                        pageid = int(pageid_str)
                        if pageid < 0:
                            continue

                        title = page_data.get("title", "")
                        content = page_data.get("extract", "")

                        if not content or not content.strip():
                            continue

                        url = f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}"

                        results[pageid] = {"title": title, "content": content, "url": url}

                    break

                except (requests.RequestException, ValueError):
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    continue

        return results

    @staticmethod
    def _has_sane_content(article_content: object) -> bool:
        """Return whether a persisted article has usable content."""
        return (
            isinstance(article_content, dict)
            and isinstance(article_content.get("content"), str)
            and bool(article_content["content"].strip())
        )

    @classmethod
    def _sanitize_existing_content(cls, existing: object) -> dict[str, ArticleContent]:
        """Normalize persisted content keys and drop unusable entries."""
        if not isinstance(existing, dict):
            return {}

        sane_content: dict[str, ArticleContent] = {}
        for pageid, article_content in existing.items():
            if cls._has_sane_content(article_content):
                sane_content[str(pageid)] = cast(ArticleContent, article_content)
        return sane_content

    @staticmethod
    def _unique_pageids(articles: list[ArticleMeta]) -> list[int]:
        """Return input page IDs once, preserving source order."""
        pageids = []
        seen = set()
        for article in articles:
            pageid = article["pageid"]
            pageid_key = str(pageid)
            if pageid_key in seen:
                continue
            seen.add(pageid_key)
            pageids.append(pageid)
        return pageids

    @staticmethod
    def _save_content(output_path: str, content: dict[str, ArticleContent]) -> None:
        write_json_atomic(output_path, content, indent=2)

    def fetch_from_file(self, input_path: str, output_path: str, batch_size: int = 50) -> None:
        """
        Read wiki_articles.json and fetch content for all articles.

        Skips:
        - Articles already in output_path (resume support)
        - Articles whose content is empty

        Args:
            input_path: Path to wiki_articles.json (must have 'pageid' field).
            output_path: Path to save article_contents.json.
            batch_size: Number of input articles to fetch before checkpointing.
        """
        existing = {}
        if os.path.exists(output_path):
            existing = read_json_file(output_path)

        articles = cast(list[ArticleMeta], read_json_file(input_path))

        pageids = self._unique_pageids(articles)
        input_pageids = {str(pid) for pid in pageids}
        all_content = self._sanitize_existing_content(existing)
        all_content = {k: v for k, v in all_content.items() if k in input_pageids}
        new_count = 0
        for i in range(0, len(pageids), batch_size):
            batch = pageids[i : i + batch_size]
            # Skip pageids already fetched with non-empty content
            batch_to_fetch = [pid for pid in batch if str(pid) not in all_content]
            if not batch_to_fetch:
                logger.info(
                    "Articles %s to %s of %s: already fetched, skipping",
                    i + 1,
                    i + len(batch),
                    len(pageids),
                )
                continue
            logger.info(
                "Fetching articles %s to %s of %s (%s new)...",
                i + 1,
                i + len(batch),
                len(pageids),
                len(batch_to_fetch),
            )
            batch_content = self.get_articles_content(batch_to_fetch)
            sane_batch_content = self._sanitize_existing_content(batch_content)
            all_content.update(sane_batch_content)
            new_count += len(sane_batch_content)

            # Checkpoint after every batch
            self._save_content(output_path, all_content)
            logger.info(
                "Checkpoint saved (%s total, %s new) - batch %s",
                len(all_content),
                new_count,
                i // batch_size + 1,
            )

        self._save_content(output_path, all_content)
        logger.info("Saved %s articles (%s new) to %s", len(all_content), new_count, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetcher = WikiContentFetcher()
    paths = DataPaths()
    fetcher.fetch_from_file(paths.wiki_articles, paths.article_contents)
