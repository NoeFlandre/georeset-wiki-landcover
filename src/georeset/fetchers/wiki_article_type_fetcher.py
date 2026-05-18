"""Fetch only Wikipedia category/page-props metadata for article types."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from georeset.analysis.article_type_classifier import ArticleTypeAssignment, assign_article_types
from georeset.config import DataPaths
from georeset.experiment_paths import experiment_artifact_file
from georeset.utils.json_io import read_json_file, write_json_atomic

logger = logging.getLogger(__name__)


class WikiArticleTypeFetcher:
    def __init__(self) -> None:
        self.api_url = "https://fr.wikipedia.org/w/api.php"
        self.headers = {
            "User-Agent": "GeoResetPipeline/1.0 (https://geo-reset.sylvainlobry.com/; research@sylvainlobry.com) python-requests/2.33.1"
        }

    def _unique_pageids(self, articles: list[dict[str, Any]]) -> list[int]:
        pageids: list[int] = []
        seen: set[str] = set()
        for article in articles:
            raw = article.get("pageid")
            if raw is None:
                continue
            pageid = str(raw)
            if pageid in seen:
                continue
            seen.add(pageid)
            pageids.append(int(pageid))
        return pageids

    @staticmethod
    def _categories_from_payload(page_payload: dict[str, Any]) -> list[str]:
        raw_categories = page_payload.get("categories", [])
        categories: list[str] = []
        for item in raw_categories:
            if isinstance(item, str):
                categories.append(item)
            elif isinstance(item, dict):
                title = item.get("title")
                if isinstance(title, str):
                    categories.append(title)
        return categories

    @staticmethod
    def _is_retryable_http_error(error: requests.HTTPError) -> bool:
        response = error.response
        if response is None:
            return False
        status_code = int(response.status_code)
        return status_code == 429 or status_code >= 500

    def _fetch_metadata_batch(
        self,
        pageids: list[int],
        *,
        max_attempts: int = 3,
        base_backoff_seconds: float = 0.2,
    ) -> dict[str, dict[str, Any]]:
        if not pageids:
            return {}
        params = {
            "action": "query",
            "prop": "categories|pageprops",
            "pageids": "|".join(str(pageid) for pageid in pageids),
            "format": "json",
            "cllimit": 500,
            "clshow": "!hidden",
        }
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    self.api_url, params=params, headers=self.headers, timeout=20
                )
                response.raise_for_status()
                payload = response.json()
                pages = payload.get("query", {}).get("pages", {})
                output: dict[str, dict[str, Any]] = {}

                for pageid, page_data in pages.items():
                    if not isinstance(page_data, dict):
                        continue
                    output[str(pageid)] = {
                        "pageid": str(page_data.get("pageid", pageid)),
                        "title": str(page_data.get("title", "")),
                        "ns": page_data.get("ns"),
                        "categories": self._categories_from_payload(page_data),
                        "pageprops": page_data.get("pageprops", {}),
                    }

                time.sleep(0.5)
                return output
            except requests.HTTPError as error:
                last_error = error
                if not self._is_retryable_http_error(error):
                    raise
            except requests.RequestException as error:
                last_error = error

            is_last_attempt = attempt >= max_attempts - 1
            if is_last_attempt:
                break
            time.sleep(base_backoff_seconds * (2 ** attempt))

        assert last_error is not None
        raise last_error

    def _pageids_to_rows(self, records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for pageid, raw_row in records.items():
            categories = raw_row.get("categories", [])
            row = assign_article_types(categories, pageid=pageid, title=str(raw_row.get("title", "")))
            rows[pageid] = self._serialize_assignment(row, raw_row)
        return rows

    @staticmethod
    def _serialize_assignment(
        assignment: ArticleTypeAssignment,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "pageid": assignment.pageid,
            "title": assignment.title,
            "ns": metadata.get("ns"),
            "pageprops": metadata.get("pageprops", {}),
            "primary_article_type": assignment.primary_article_type,
            "candidate_article_types": assignment.candidate_article_types,
            "matched_categories": assignment.matched_categories,
            "matched_rules": assignment.matched_rules,
            "all_categories_count": assignment.all_categories_count,
            "has_categories": assignment.has_categories,
        }

    @staticmethod
    def _prune_stale_records(records: dict[str, Any], pageids: set[str]) -> dict[str, dict[str, Any]]:
        return {
            pageid: value
            for pageid, value in records.items()
            if isinstance(value, dict) and str(pageid) in pageids
        }

    @staticmethod
    def _sanitize_existing(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(pageid): value
            for pageid, value in records.items()
            if isinstance(pageid, (str, int)) and isinstance(value, dict)
            and str(pageid) not in {"", None}
            and isinstance(value.get("primary_article_type"), str)
        }

    def fetch_from_file(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        batch_size: int = 50,
    ) -> None:
        existing = {}
        if os.path.exists(output_path):
            existing = read_json_file(output_path)

        articles = read_json_file(input_path)
        if not isinstance(articles, list):
            return

        pageids = self._unique_pageids(articles)
        input_pageids = {str(pageid) for pageid in pageids}
        all_rows = self._sanitize_existing(existing)
        all_rows = self._prune_stale_records(all_rows, input_pageids)

        try:
            for start in range(0, len(pageids), batch_size):
                batch = pageids[start : start + batch_size]
                batch_to_fetch = [pageid for pageid in batch if str(pageid) not in all_rows]
                if not batch_to_fetch:
                    continue
                raw_metadata = self._fetch_metadata_batch(batch_to_fetch)
                rows = self._pageids_to_rows(raw_metadata)
                all_rows.update(rows)
                write_json_atomic(output_path, all_rows, indent=2, ensure_ascii=False)
        except KeyboardInterrupt:
            write_json_atomic(output_path, all_rows, indent=2, ensure_ascii=False)
            raise

        write_json_atomic(output_path, all_rows, indent=2, ensure_ascii=False)


def main() -> None:
    fetcher = WikiArticleTypeFetcher()
    data_paths = DataPaths()
    fetcher.fetch_from_file(
        data_paths.wiki_articles,
        experiment_artifact_file(
            "article_text_classification_article_type_relevance_stratified_v1",
            "article_type_metadata.json",
        ),
    )


if __name__ == "__main__":
    main()
