"""Tests for article summarizer."""

import json
import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock

from src.scripts.summarize_articles import ArticleSummarizer


class TestArticleSummarizer:
    def setup_method(self):
        self.sample_article = {
            "title": "Strasbourg",
            "content": "Strasbourg est une ville française, préfecture du Bas-Rhin. Elle est connue pour sa cathédrale gothique.",
            "url": "https://fr.wikipedia.org/wiki/Strasbourg"
        }

    def test_summarize_returns_summary(self):
        """Should return a dict with summary key for an article."""
        summarizer = ArticleSummarizer(model_path=None)  # Will be mocked
        with patch.object(summarizer, "_call_llm", return_value="Une ville française."):
            result = summarizer.summarize(self.sample_article)
            assert isinstance(result, dict)
            assert "summary" in result
            assert result["summary"] == "Une ville française."

    def test_summarize_adds_summary_key(self):
        """Should add summary key to article dict without modifying original."""
        summarizer = ArticleSummarizer(model_path=None)
        with patch.object(summarizer, "_call_llm", return_value="Résumé test"):
            article_copy = dict(self.sample_article)
            result = summarizer.summarize(article_copy)
            assert "summary" in result
            assert "title" in result
            assert "content" in result

    def test_summarize_preserves_original_fields(self):
        """Should not modify title, content, or url."""
        summarizer = ArticleSummarizer(model_path=None)
        with patch.object(summarizer, "_call_llm", return_value="Résumé"):
            result = summarizer.summarize(self.sample_article)
            assert result["title"] == self.sample_article["title"]
            assert result["content"] == self.sample_article["content"]
            assert result["url"] == self.sample_article["url"]

    def test_process_file_skips_existing_and_saves_progress(self):
        """Should skip already-summarized articles and save progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "articles.json")
            output_path = os.path.join(tmpdir, "summaries.json")

            articles = {
                "1": {"title": "A", "content": "Content A", "url": "http://a"},
                "2": {"title": "B", "content": "Content B", "url": "http://b"},
            }
            with open(input_path, "w") as f:
                json.dump(articles, f)

            summarizer = ArticleSummarizer(model_path=None)
            with patch.object(summarizer, "_call_llm", side_effect=["Summary A", "Summary B"]):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)

            assert len(result) == 2
            assert result["1"]["summary"] == "Summary A"
            assert result["2"]["summary"] == "Summary B"

    def test_process_file_resumes_from_checkpoint(self):
        """Should skip articles that already have a summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "articles.json")
            output_path = os.path.join(tmpdir, "summaries.json")

            articles = {
                "1": {"title": "A", "content": "Content A", "url": "http://a"},
                "2": {"title": "B", "content": "Content B", "url": "http://b"},
            }
            with open(input_path, "w") as f:
                json.dump(articles, f)

            # Pre-existing output with article 1 already summarized
            existing = {
                "1": {"title": "A", "content": "Content A", "url": "http://a", "summary": "Old summary"}
            }
            with open(output_path, "w") as f:
                json.dump(existing, f)

            summarizer = ArticleSummarizer(model_path=None)

            with patch.object(summarizer, "_call_llm", return_value=f"New summary for B"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert result["1"]["summary"] == "Old summary"  # Preserved
            assert result["2"]["summary"] == "New summary for B"  # Added

    def test_process_file_handles_empty_content(self):
        """Should handle articles with empty content gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "articles.json")
            output_path = os.path.join(tmpdir, "summaries.json")

            articles = {
                "1": {"title": "Empty", "content": "", "url": "http://empty"},
            }
            with open(input_path, "w") as f:
                json.dump(articles, f)

            summarizer = ArticleSummarizer(model_path=None)
            with patch.object(summarizer, "_call_llm", return_value="Summary of empty"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert "summary" in result["1"]
