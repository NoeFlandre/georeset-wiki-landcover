"""Tests for land-use evidence summarization CLI."""

import json
from collections.abc import Iterator
from pathlib import Path

from georeset.cli.data.summarize_landuse_evidence import main, parse_args
from georeset.config import DataPaths, ModelSettings
from georeset.fetchers.landuse_evidence_summarizer import LandUseEvidenceSummarizer


class _FakeSummarizer:
    """Capture constructor args and run behavior for CLI-level tests."""

    def __init__(
        self,
        model_path: str,
        model_repo_id: str | None,
        seed: int,
        temperature: float,
    ) -> None:
        self.model_path = model_path
        self.model_repo_id = model_repo_id
        self.seed = seed
        self.temperature = temperature
        self.process_calls: list[tuple[str, str]] = []

    def process_file(self, input_path: str, output_path: str) -> None:
        self.process_calls.append((input_path, output_path))
        # Write an empty valid artifact to emulate resumable checkpoint behavior.
        Path(output_path).write_text("{}", encoding="utf-8")


def _fake_client_response() -> Iterator[dict[str, str]]:
    yield {
        "landuse_evidence_summary": "One zone de culture et d'eau est mentionnée.",
        "landcover_relevance": "low",
        "evidence_types": ["water", "agriculture"],
        "evidence_sentences_no_place": ["Rivières et cultures.", "Culture légère."],
        "uncertainty": "low",
    }


def test_parse_args_defaults_and_new_flags(monkeypatch):
    monkeypatch.delenv("GEORESET_MODEL_PATH", raising=False)

    args = parse_args([])

    assert args.input_path == DataPaths().article_contents
    assert args.output_path == DataPaths().article_landuse_evidence_summaries
    assert args.model_path == ModelSettings().model_path
    assert args.seed == ModelSettings().seed
    assert args.temperature == 0.0
    assert args.model_repo_id is None


def test_parse_args_accepts_overrides(monkeypatch):
    monkeypatch.setenv("GEORESET_MODEL_PATH", "custom.gguf")
    args = parse_args(["--model-repo-id", "org/repo", "--seed", "13", "--temperature", "0.25"])

    assert args.model_path == "custom.gguf"
    assert args.model_repo_id == "org/repo"
    assert args.seed == 13
    assert args.temperature == 0.25


def test_main_invokes_landuse_summarizer_with_expected_args(monkeypatch, tmp_path):
    summary_path = tmp_path / "articles.json"
    output_path = tmp_path / "evidence.json"
    summary_path.write_text(
        json.dumps(
            {
                "1": {
                    "title": "Ville",
                    "content": "La ville contient des vignes et des zones humides.",
                    "url": "https://fr.wikipedia.org/wiki/Ville",
                }
            }
        ),
        encoding="utf-8",
    )

    recorded = {}

    def _factory(**kwargs: object) -> _FakeSummarizer:
        summarizer = _FakeSummarizer(
            model_path=kwargs["model_path"],
            model_repo_id=kwargs.get("model_repo_id"),
            seed=kwargs["seed"],
            temperature=kwargs["temperature"],
        )
        recorded["summarizer"] = summarizer
        return summarizer

    monkeypatch.setattr(
        "georeset.cli.data.summarize_landuse_evidence.LandUseEvidenceSummarizer", _factory
    )
    main(
        [
            "--input-path",
            str(summary_path),
            "--output-path",
            str(output_path),
            "--model-path",
            "qwen.gguf",
            "--model-repo-id",
            "hf/repo",
            "--seed",
            "7",
            "--temperature",
            "0.05",
        ]
    )

    summarizer = recorded["summarizer"]
    assert summarizer.model_path == "qwen.gguf"
    assert summarizer.model_repo_id == "hf/repo"
    assert summarizer.seed == 7
    assert summarizer.temperature == 0.05
    assert summarizer.process_calls == [(str(summary_path), str(output_path))]


def test_process_file_resumes_valid_and_regenerates_malformed_records(tmp_path):
    input_path = tmp_path / "article_contents.json"
    output_path = tmp_path / "article_landuse_evidence_summaries.json"
    input_path.write_text(
        json.dumps(
            {
                "1": {
                    "title": "Nice",
                    "content": "La commune est près d'une rivière et de vignobles.",
                    "url": "https://fr.wikipedia.org/wiki/Nice",
                },
                "2": {
                    "title": "Lyon",
                    "content": "Lyon est une grande ville.",
                    "url": "https://fr.wikipedia.org/wiki/Lyon",
                },
            }
        ),
        encoding="utf-8",
    )

    valid_record = {
        "title": "Nice",
        "content": "La commune est près d'une rivière et de vignobles.",
        "url": "https://fr.wikipedia.org/wiki/Nice",
        "landuse_evidence_summary": "La zone comprend des vignes et des cours d'eau.",
        "landcover_relevance": "medium",
        "evidence_types": ["water", "agriculture"],
        "evidence_sentences_no_place": ["Zone de vignes en bordure de rivière."],
        "uncertainty": "low",
        "landuse_evidence_summary_char_count": 47,
        "evidence_sentences_count": 1,
        "metadata": {
            "evidence_mode": "landuse_evidence",
            "prompt_version": 1,
            "summary_no_place": True,
            "model": "Qwen3.6-27B-Q4_0.gguf",
            "model_repo_id": None,
            "seed": 42,
            "temperature": 0.0,
            "prompt": "prompt",
            "system_prompt": "system",
        },
    }
    malformed_record = {
        "title": "Lyon",
        "content": "Lyon est une grande ville.",
        "url": "https://fr.wikipedia.org/wiki/Lyon",
        "landuse_evidence_summary": "bad",
        "landcover_relevance": "invalid",
        "evidence_types": ["water"],
        "evidence_sentences_no_place": ["ville"],
        "uncertainty": "low",
        "landuse_evidence_summary_char_count": 3,
        "evidence_sentences_count": 1,
    }
    output_path.write_text(
        json.dumps(
            {"1": valid_record, "2": malformed_record, "3": {"stale": "entry"}}, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    response = json.dumps(next(_fake_client_response()))
    summarizer = LandUseEvidenceSummarizer(model_path="Qwen3.6-27B-Q4_0.gguf", client=None)

    # inject deterministic LLM response path
    summarizer._client = type("C", (), {"complete_json": lambda self, **_: response})()

    summarizer.process_file(str(input_path), str(output_path))

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(result) == {"1", "2"}
    assert result["1"]["landuse_evidence_summary"] == valid_record["landuse_evidence_summary"]
    assert result["2"]["landuse_evidence_summary"] == "One zone de culture et d'eau est mentionnée."
    assert result["2"]["landuse_evidence_summary_char_count"] == len(
        result["2"]["landuse_evidence_summary"]
    )
    assert result["2"]["evidence_sentences_count"] == 2


def test_main_does_not_touch_frozen_experiment_directories(tmp_path, monkeypatch):
    input_path = tmp_path / "articles.json"
    output_path = tmp_path / "evidence.json"
    input_path.write_text("{}", encoding="utf-8")

    frozen_dir = tmp_path / "data" / "experiments" / "frozen"
    frozen_file = frozen_dir / "ignore.json"
    frozen_dir.mkdir(parents=True)
    frozen_file.write_text('{"run": "keep"}', encoding="utf-8")

    fake = _FakeSummarizer("Qwen3.6-27B-Q4_0.gguf", None, 42, 0.0)

    def _factory(**_: object) -> _FakeSummarizer:
        return fake

    monkeypatch.setattr(
        "georeset.cli.data.summarize_landuse_evidence.LandUseEvidenceSummarizer", _factory
    )
    main(["--input-path", str(input_path), "--output-path", str(output_path)])

    assert output_path.exists()
    assert frozen_file.exists()
    assert frozen_file.read_text(encoding="utf-8") == '{"run": "keep"}'
