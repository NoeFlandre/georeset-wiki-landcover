"""Tests for article summarizer."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.data.summarize_articles import parse_args
from src.fetchers.article_summarizer import ArticleSummarizer


class TestArticleSummarizer:
    def setup_method(self):
        self.sample_article = {
            "title": "Strasbourg",
            "content": "Strasbourg est une ville française, préfecture du Bas-Rhin. Elle est connue pour sa cathédrale gothique.",
            "url": "https://fr.wikipedia.org/wiki/Strasbourg",
        }

    def test_summarize_returns_summary_and_metadata(self):
        """Should return a dict with summary key and metadata for an article."""
        summarizer = ArticleSummarizer(model_path="test_model.gguf", seed=123, temperature=0.5)
        expected_prompt = (
            f"Résumez cet article Wikipedia en une phrase concise:\n\n"
            f"{self.sample_article['content']}"
        )
        with patch.object(summarizer, "_generate_summary", return_value="Une ville française."):
            result = summarizer.summarize(self.sample_article)
            assert isinstance(result, dict)
            assert "summary" in result
            assert result["summary"] == "Une ville française."
            assert "thinking" not in result
            assert "metadata" in result
            assert result["metadata"]["model"] == "test_model.gguf"
            assert result["metadata"]["seed"] == 123
            assert result["metadata"]["temperature"] == 0.5
            assert result["metadata"]["summary_mode"] == "place"
            assert result["metadata"]["prompt"] == expected_prompt

    def test_no_place_mode_uses_no_place_prompt_and_metadata(self):
        summarizer = ArticleSummarizer(model_path=None, summary_mode="no_place")

        with patch.object(summarizer, "_generate_summary", return_value="Une ville française."):
            result = summarizer.summarize(self.sample_article)

        assert "sans jamais mentionner le nom du lieu décrit" in result["metadata"]["prompt"]
        assert result["metadata"]["summary_mode"] == "no_place"

    def test_invalid_summary_mode_fails_fast(self):
        with pytest.raises(ValueError, match="summary_mode"):
            ArticleSummarizer(model_path=None, summary_mode="invalid")

    def test_gpu_optimization_enabled(self):
        """Should initialize Llama with n_gpu_layers=-1 for GPU acceleration."""
        summarizer = ArticleSummarizer(model_path="test_model.gguf")
        mock_llama_cpp = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            summarizer._get_llm()
            mock_llama_cpp.Llama.from_pretrained.assert_called_once()
            _, kwargs = mock_llama_cpp.Llama.from_pretrained.call_args
            assert kwargs.get("n_gpu_layers") == -1

    def test_summarize_adds_summary_key(self):
        """Should add summary key to article dict without modifying original."""
        summarizer = ArticleSummarizer(model_path=None)
        with patch.object(summarizer, "_generate_summary", return_value="Résumé test"):
            article_copy = dict(self.sample_article)
            result = summarizer.summarize(article_copy)
            assert "summary" in result
            assert "title" in result
            assert "content" in result

    def test_summarize_preserves_original_fields(self):
        """Should not modify title, content, or url."""
        summarizer = ArticleSummarizer(model_path=None)
        with patch.object(summarizer, "_generate_summary", return_value="Résumé"):
            result = summarizer.summarize(self.sample_article)
            assert result["title"] == self.sample_article["title"]
            assert result["content"] == self.sample_article["content"]
            assert result["url"] == self.sample_article["url"]

    def test_generate_summary_uses_llama_cpp_json_schema_mode(self):
        """Should constrain llama.cpp output to the summary schema."""
        mock_client = MagicMock()
        mock_client.complete_json.return_value = '{"summary": "Une ville française."}'
        summarizer = ArticleSummarizer(
            model_path=None, seed=123, temperature=0.5, client=mock_client
        )

        assert summarizer._generate_summary("Prompt", "System prompt") == "Une ville française."

        _, kwargs = mock_client.complete_json.call_args
        assert kwargs["schema"] == ArticleSummarizer.SUMMARY_SCHEMA
        assert kwargs["max_tokens"] == 256
        assert "thinking" not in json.dumps(kwargs["schema"])

    def test_generate_summary_rejects_thinking_polluted_response(self):
        """Should fail loudly if the backend does not honor structured output."""
        summarizer = ArticleSummarizer(model_path=None)
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '<think>hidden</think>{"summary": "Résumé"}'}}]
        }
        summarizer._llm = mock_llm

        with pytest.raises(ValueError, match="valid summary JSON"):
            summarizer._generate_summary("Prompt", "System prompt")

    def test_generate_summary_rejects_private_markers_inside_summary(self):
        """Should not persist summaries that contain private reasoning markers."""
        summarizer = ArticleSummarizer(model_path=None)

        with pytest.raises(ValueError, match="private thinking markers"):
            summarizer._summary_from_response('{"summary": "<think>hidden</think> Résumé"}')

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
            with patch.object(
                summarizer, "_generate_summary", side_effect=["Summary A", "Summary B"]
            ):
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
                "1": {
                    "title": "A",
                    "content": "Content A",
                    "url": "http://a",
                    "summary": "Old summary",
                    "metadata": {
                        "prompt": "Résumez cet article Wikipedia en une phrase concise:\n\nContent A"
                    },
                }
            }
            with open(output_path, "w") as f:
                json.dump(existing, f)

            summarizer = ArticleSummarizer(model_path=None)

            with patch.object(summarizer, "_generate_summary", return_value="New summary for B"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert result["1"]["summary"] == "Old summary"  # Preserved
            assert result["2"]["summary"] == "New summary for B"  # Added

    def test_process_file_reprocesses_existing_summary_from_wrong_mode(self):
        """Existing summaries must not be reused across place/no_place modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "articles.json")
            output_path = os.path.join(tmpdir, "summaries.json")

            article = {"title": "A", "content": "Content A", "url": "http://a"}
            with open(input_path, "w") as f:
                json.dump({"1": article}, f)

            with open(output_path, "w") as f:
                json.dump(
                    {
                        "1": {
                            **article,
                            "summary": "Old no-place summary",
                            "metadata": {"summary_mode": "no_place"},
                        }
                    },
                    f,
                )

            summarizer = ArticleSummarizer(model_path=None, summary_mode="place")
            with patch.object(summarizer, "_generate_summary", return_value="New place summary"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert result["1"]["summary"] == "New place summary"
            assert result["1"]["metadata"]["summary_mode"] == "place"

    def test_process_file_removes_existing_private_fields(self):
        """Should clean private fields from resumed output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "articles.json")
            output_path = os.path.join(tmpdir, "summaries.json")

            article = {"title": "A", "content": "Content A", "url": "http://a"}
            with open(input_path, "w") as f:
                json.dump({"1": article}, f)

            with open(output_path, "w") as f:
                json.dump({"1": {**article, "summary": "Old summary", "thinking": "private"}}, f)

            summarizer = ArticleSummarizer(model_path=None)
            with patch.object(summarizer, "_generate_summary", return_value="New summary"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert result["1"]["summary"] == "New summary"
            assert "thinking" not in result["1"]

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
            with patch.object(summarizer, "_generate_summary", return_value="Summary of empty"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert "summary" in result["1"]

    def test_process_file_prunes_stale_pageids(self):
        """Existing summary key absent from current contents must be dropped on resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "articles.json")
            output_path = os.path.join(tmpdir, "summaries.json")

            with open(input_path, "w") as f:
                json.dump({"100": {"title": "A", "content": "Content A", "url": "http://a"}}, f)

            with open(output_path, "w") as f:
                json.dump(
                    {
                        "100": {
                            "title": "A",
                            "content": "Content A",
                            "url": "http://a",
                            "summary": "Old",
                            "metadata": {
                                "prompt": "Résumez cet article Wikipedia en une phrase concise:\n\nContent A"
                            },
                        },
                        "999": {
                            "title": "Stale",
                            "content": "Stale",
                            "url": "http://stale",
                            "summary": "StaleSum",
                        },
                    },
                    f,
                )

            summarizer = ArticleSummarizer(model_path=None)
            with patch.object(summarizer, "_generate_summary", return_value="New"):
                summarizer.process_file(input_path, output_path)

            with open(output_path) as f:
                result = json.load(f)
            assert "999" not in result
            assert "100" in result
            assert result["100"]["summary"] == "Old"


def test_parse_args_uses_grid5000_ready_defaults(monkeypatch):
    """Should default to the expected resumable summarization files."""
    monkeypatch.delenv("GEORESET_MODEL_PATH", raising=False)

    args = parse_args([])

    assert args.input_path == "data/wiki/article_contents.json"
    assert args.output_path == "data/wiki/article_summaries.json"
    assert args.summary_mode == "place"
    assert args.model_path == "Qwen3.6-27B-Q4_0.gguf"
    assert args.seed == 42
    assert args.temperature == 0.7


def test_parse_args_allows_environment_model_override(monkeypatch):
    """Should let Grid5000 jobs select a model without patching Python code."""
    monkeypatch.setenv("GEORESET_MODEL_PATH", "custom.gguf")

    args = parse_args([])

    assert args.model_path == "custom.gguf"


def test_parse_args_accepts_no_place_summary_mode():
    args = parse_args(["--summary-mode", "no_place"])

    assert args.summary_mode == "no_place"
