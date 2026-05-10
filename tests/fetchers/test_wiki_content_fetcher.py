"""Tests for WikiContentFetcher."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.fetchers.wiki_content_fetcher import WikiContentFetcher


class TestWikiContentFetcher:
    def setup_method(self):
        self.fetcher = WikiContentFetcher()

    @patch("requests.get")
    def test_fetch_from_file_skips_existing_content(self, mock_get):
        """Should not re-fetch pageids that already have content in output."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "pages": {
                    "123": {"pageid": 123, "title": "AlreadyFetched", "extract": "Old content"},
                    "456": {"pageid": 456, "title": "NewArticle", "extract": "New content"},
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            existing = {
                "123": {
                    "title": "AlreadyFetched",
                    "content": "Old content",
                    "url": "https://fr.wikipedia.org/wiki/AlreadyFetched",
                }
            }
            json.dump(existing, f)
            output_path = f.name

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                articles = [{"pageid": 123}, {"pageid": 456}]
                json.dump(articles, f)
                input_path = f.name

            self.fetcher.fetch_from_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)

            assert "123" in result
            assert result["123"]["content"] == "Old content"
            assert "456" in result
            assert result["456"]["content"] == "New content"
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    @patch("requests.get")
    def test_fetch_from_file_article_count_matches_content_count(self, mock_get):
        """Number of fetched articles should equal number of content entries (minus negatives)."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "pages": {
                    "100": {"pageid": 100, "title": "A", "extract": "Content A"},
                    "200": {"pageid": 200, "title": "B", "extract": "Content B"},
                    "300": {"pageid": 300, "title": "C", "extract": "Content C"},
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            output_path = f.name

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                articles = [{"pageid": 100}, {"pageid": 200}, {"pageid": 300}]
                json.dump(articles, f)
                input_path = f.name

            self.fetcher.fetch_from_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)

            # Should have 3 entries (one per valid pageid)
            assert len(result) == 3
            assert result["100"]["content"] == "Content A"
            assert result["200"]["content"] == "Content B"
            assert result["300"]["content"] == "Content C"
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    @patch("requests.get")
    def test_fetch_from_file_handles_missing_pages(self, mock_get):
        """Should skip negative pageids (pages that don't exist) without affecting count."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "pages": {
                    "-1": {"title": "Missing", "missing": ""},
                    "100": {"pageid": 100, "title": "Valid", "extract": "Content"},
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            output_path = f.name

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                articles = [{"pageid": 999}, {"pageid": 100}]  # 999 doesn't exist
                json.dump(articles, f)
                input_path = f.name

            self.fetcher.fetch_from_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)

            # Only the valid pageid should be in result
            assert len(result) == 1
            assert "100" in result
            assert "999" not in result
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    @patch("requests.get")
    def test_get_articles_content_uses_full_page_not_just_intro(self, mock_get):
        """Should request full article content, not just intro (no exintro param)."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "pages": {"100": {"pageid": 100, "title": "Test", "extract": "Full content here"}}
            }
        }

        self.fetcher.get_articles_content([100])

        # Verify exintro was NOT in the params
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert "exintro" not in params

    @patch("requests.get")
    def test_fetch_from_file_drops_articles_with_empty_content(self, mock_get):
        """Should not include articles where content is empty or whitespace only."""
        mock_get.return_value.headers = {"Content-Type": "application/json"}
        mock_get.return_value.json.return_value = {
            "query": {
                "pages": {
                    "100": {"pageid": 100, "title": "Valid", "extract": "Has content"},
                    "200": {"pageid": 200, "title": "Empty", "extract": ""},
                    "300": {"pageid": 300, "title": "Whitespace", "extract": "   "},
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            output_path = f.name

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                articles = [{"pageid": 100}, {"pageid": 200}, {"pageid": 300}]
                json.dump(articles, f)
                input_path = f.name

            self.fetcher.fetch_from_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)

            # Only article 100 should be kept (has actual content)
            assert len(result) == 1
            assert "100" in result
            assert "200" not in result  # empty string
            assert "300" not in result  # whitespace only
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    def test_fetch_from_file_sanitizes_existing_output_before_resuming(self):
        """Should drop corrupt existing entries and re-fetch them on resume."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            existing = {
                "100": {
                    "title": "Valid",
                    "content": "Existing content",
                    "url": "https://fr.wikipedia.org/wiki/Valid",
                },
                "200": {
                    "title": "Empty",
                    "content": "   ",
                    "url": "https://fr.wikipedia.org/wiki/Empty",
                },
                "300": {
                    "title": "MissingContent",
                    "url": "https://fr.wikipedia.org/wiki/MissingContent",
                },
            }
            json.dump(existing, f)
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            articles = [{"pageid": 100}, {"pageid": 200}, {"pageid": 300}]
            json.dump(articles, f)
            input_path = f.name

        try:
            with patch.object(
                self.fetcher,
                "get_articles_content",
                return_value={
                    "200": {
                        "title": "Empty",
                        "content": "Fresh content",
                        "url": "https://fr.wikipedia.org/wiki/Empty",
                    },
                    "300": {
                        "title": "MissingContent",
                        "content": "Recovered content",
                        "url": "https://fr.wikipedia.org/wiki/MissingContent",
                    },
                },
            ) as get_articles_content:
                self.fetcher.fetch_from_file(input_path, output_path)

            get_articles_content.assert_called_once_with([200, 300])
            with open(output_path) as f:
                result = json.load(f)

            assert result == {
                "100": {
                    "title": "Valid",
                    "content": "Existing content",
                    "url": "https://fr.wikipedia.org/wiki/Valid",
                },
                "200": {
                    "title": "Empty",
                    "content": "Fresh content",
                    "url": "https://fr.wikipedia.org/wiki/Empty",
                },
                "300": {
                    "title": "MissingContent",
                    "content": "Recovered content",
                    "url": "https://fr.wikipedia.org/wiki/MissingContent",
                },
            }
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    def test_fetch_from_file_deduplicates_input_pageids(self):
        """Should fetch duplicate page IDs once and write one content entry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            articles = [{"pageid": 100}, {"pageid": 100}, {"pageid": 200}]
            json.dump(articles, f)
            input_path = f.name

        try:
            with patch.object(
                self.fetcher,
                "get_articles_content",
                return_value={
                    "100": {
                        "title": "A",
                        "content": "Content A",
                        "url": "https://fr.wikipedia.org/wiki/A",
                    },
                    "200": {
                        "title": "B",
                        "content": "Content B",
                        "url": "https://fr.wikipedia.org/wiki/B",
                    },
                },
            ) as get_articles_content:
                self.fetcher.fetch_from_file(input_path, output_path, batch_size=2)

            get_articles_content.assert_called_once_with([100, 200])
            with open(output_path) as f:
                result = json.load(f)

            assert list(result) == ["100", "200"]
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    def test_fetch_from_file_saves_progress_when_stopped(self):
        """Should persist fetched content before propagating a stop signal."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            articles = [{"pageid": 100}, {"pageid": 200}, {"pageid": 300}]
            json.dump(articles, f)
            input_path = f.name

        try:
            with (
                patch.object(
                    self.fetcher,
                    "get_articles_content",
                    side_effect=[
                        {
                            "100": {
                                "title": "A",
                                "content": "Content A",
                                "url": "https://fr.wikipedia.org/wiki/A",
                            }
                        },
                        KeyboardInterrupt,
                    ],
                ),
                pytest.raises(KeyboardInterrupt),
            ):
                self.fetcher.fetch_from_file(input_path, output_path, batch_size=1)

            with open(output_path) as f:
                result = json.load(f)

            assert result == {
                "100": {
                    "title": "A",
                    "content": "Content A",
                    "url": "https://fr.wikipedia.org/wiki/A",
                }
            }
        finally:
            os.unlink(output_path)
            os.unlink(input_path)

    def test_fetch_from_file_prunes_stale_pageids(self):
        """Existing output pageid absent from current input must be dropped on resume."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "999": {"title": "Stale", "content": "Old", "url": "http://stale"},
                    "100": {"title": "Valid", "content": "Valid", "url": "http://valid"},
                },
                f,
            )
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"pageid": 100}], f)
            input_path = f.name

        try:
            with patch.object(self.fetcher, "get_articles_content", return_value={}):
                self.fetcher.fetch_from_file(input_path, output_path)
            with open(output_path) as f:
                result = json.load(f)
            assert "999" not in result
            assert "100" in result
        finally:
            os.unlink(output_path)
            os.unlink(input_path)
