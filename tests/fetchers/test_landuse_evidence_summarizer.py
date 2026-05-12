"""Tests for land-use evidence summarization."""

import json
from typing import Any

import pytest

from georeset.config import DataPaths
from georeset.fetchers.landuse_evidence_summarizer import (
    EVIDENCE_TYPES,
    LANDUSE_EVIDENCE_PROMPT_VERSION,
    LandUseEvidenceSummarizer,
)

ARTICLE = {
    "title": "La-Sainte_Marie",
    "content": "La ville de La-Sainte_Marie est connue pour ses vignes et ses forêts.",
    "url": "https://fr.wikipedia.org/wiki/La-Sainte_Marie",
}


class _FakeClient:
    def __init__(self, response: str | list[str]):
        if isinstance(response, str):
            self.responses = [response]
        else:
            self.responses = list(response)
        self.calls: list[tuple[str, str]] = []
        self.call_index = 0

    def complete_json(self, **kwargs: Any) -> str:
        self.calls.append((kwargs["system_prompt"], kwargs["user_prompt"]))
        response = self.responses[min(self.call_index, len(self.responses) - 1)]
        self.call_index += 1
        return response


@pytest.fixture()
def fake_good_response() -> str:
    return json.dumps(
        {
            "landuse_evidence_summary": "Une zone boisée et des vignes sont mentionnées. ",
            "landcover_relevance": "medium",
            "evidence_types": ["forest", "vineyard"],
            "evidence_sentences_no_place": [
                " Une grande forêt mixte couvre le nord de la commune. ",
                "des vignes historiques forment une ceinture.",
            ],
            "uncertainty": "low",
        }
    )


def _build_summarizer(response: str) -> tuple[LandUseEvidenceSummarizer, _FakeClient]:
    client = _FakeClient(response=response)
    return LandUseEvidenceSummarizer(model_path="test_model.gguf", client=client), client


def test_schema_and_prompt_have_fixed_version() -> None:
    assert LANDUSE_EVIDENCE_PROMPT_VERSION == 2
    assert (
        LandUseEvidenceSummarizer.EVIDENCE_SCHEMA["properties"]["evidence_types"]["items"][
            "enum"
        ]
        == list(EVIDENCE_TYPES)
    )


def test_summarize_adds_required_evidence_fields_and_counts(fake_good_response: str) -> None:
    summarizer, _ = _build_summarizer(fake_good_response)

    result = summarizer.summarize(ARTICLE)

    assert result["landuse_evidence_summary"] == "Une zone boisée et des vignes sont mentionnées."
    assert result["landcover_relevance"] == "medium"
    assert result["evidence_types"] == ["forest", "vineyard"]
    assert result["evidence_sentences_no_place"] == [
        "Une grande forêt mixte couvre le nord de la commune.",
        "des vignes historiques forment une ceinture.",
    ]
    assert result["uncertainty"] == "low"
    assert result["landuse_evidence_summary_char_count"] == len(result["landuse_evidence_summary"])
    assert result["evidence_sentences_count"] == 2
    metadata = result["metadata"]
    assert metadata["model"] == "test_model.gguf"
    assert metadata["seed"] == 42
    assert metadata["temperature"] == 0.0
    assert metadata["evidence_mode"] == "landuse_evidence"
    assert metadata["prompt_version"] == LANDUSE_EVIDENCE_PROMPT_VERSION
    assert metadata["summary_no_place"] is True
    assert metadata["attempt_count"] == 1
    assert "prompt" in metadata
    assert "system_prompt" in metadata


def test_prompt_demands_no_place_output(fake_good_response: str) -> None:
    _, client = _build_summarizer(fake_good_response)
    summarizer = LandUseEvidenceSummarizer(model_path="m", client=client)
    summarizer.summarize(ARTICLE)

    system_prompt, user_prompt = client.calls[0]
    assert "assistant d'extraction d'indices d'occupation du sol" in system_prompt
    assert "Ta tâche n'est pas de résumer l'article en général" in system_prompt
    assert "Titre de l'article à ne jamais mentionner ni utiliser comme indice" in user_prompt
    assert "Titre à ne jamais mentionner:" not in user_prompt
    assert "Texte de l'article:" in user_prompt
    assert "Résumé de l'article:" not in user_prompt
    assert "N'utilise pas le titre comme preuve." in user_prompt
    assert "Ne mentionne jamais le titre, le nom du lieu décrit" in user_prompt
    assert "Ignore les informations non pertinentes: histoire, dates, monuments" in user_prompt
    assert "Ne déduis rien qui n'est pas explicitement présent" in user_prompt
    assert "Reformule les preuves en français clair" in user_prompt
    assert "Retourne exactement cet objet JSON" in user_prompt
    assert "ne doit mentionner ni la source ni le titre" not in user_prompt
    assert "landuse_evidence_summary" in user_prompt
    assert "evidence_sentences_no_place" in user_prompt
    assert "no relevant evidence" not in system_prompt.lower()


def test_parse_rejects_malformed_json(fake_good_response: str) -> None:
    summarizer, _ = _build_summarizer("{not json")

    with pytest.raises(ValueError, match="valid land-use evidence JSON"):
        summarizer.summarize(ARTICLE)


def test_invalid_max_attempts_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        LandUseEvidenceSummarizer(model_path="test_model.gguf", max_attempts=0)


def test_retry_retried_once_after_malformed_json(fake_good_response: str) -> None:
    responses = ["{not json", fake_good_response]
    client = _FakeClient(responses)
    summarizer = LandUseEvidenceSummarizer(model_path="test_model.gguf", client=client)

    result = summarizer.summarize(ARTICLE)

    assert result["landuse_evidence_summary"] == "Une zone boisée et des vignes sont mentionnées."
    assert result["metadata"]["attempt_count"] == 2
    assert result["metadata"]["prompt"] == client.calls[1][1]
    assert "Correction" in result["metadata"]["prompt"]
    assert len(client.calls) == 2


def test_retry_fails_after_max_attempts() -> None:
    responses = [
        "{not json",
        json.dumps(
            {
                "landuse_evidence_summary": "L'article évoque des vignes.",
                "landcover_relevance": "unknown",
                "evidence_types": ["forest"],
                "evidence_sentences_no_place": ["Des vignes sont mentionnées."],
                "uncertainty": "low",
            }
        ),
    ]
    client = _FakeClient(responses)
    summarizer = LandUseEvidenceSummarizer(model_path="test_model.gguf", client=client)

    with pytest.raises(ValueError, match="invalid landcover_relevance"):
        summarizer.summarize(ARTICLE)

    assert len(client.calls) == 2


def test_parse_rejects_missing_fields(fake_good_response: str) -> None:
    payload = json.loads(fake_good_response)
    payload.pop("landcover_relevance")
    summarizer, _ = _build_summarizer(json.dumps(payload))

    with pytest.raises(ValueError, match="missing required fields"):
        summarizer.summarize(ARTICLE)


def test_parse_rejects_invalid_enums(fake_good_response: str) -> None:
    payload = json.loads(fake_good_response)
    payload["landcover_relevance"] = "unknown"
    summarizer, _ = _build_summarizer(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid landcover_relevance"):
        summarizer.summarize(ARTICLE)

    payload = json.loads(fake_good_response)
    payload["uncertainty"] = "unknown"
    summarizer, _ = _build_summarizer(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid uncertainty"):
        summarizer.summarize(ARTICLE)


def test_parse_rejects_bad_evidence_types(fake_good_response: str) -> None:
    payload = json.loads(fake_good_response)
    payload["evidence_types"] = ["forest", "invalid-type"]
    summarizer, _ = _build_summarizer(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid evidence_types"):
        summarizer.summarize(ARTICLE)


def test_parse_rejects_non_list_fields() -> None:
    summarizer, _ = _build_summarizer(
        json.dumps(
            {
                "landuse_evidence_summary": "Summary",
                "landcover_relevance": "none",
                "evidence_types": "forest",
                "evidence_sentences_no_place": "sentence",
                "uncertainty": "low",
            }
        )
    )

    with pytest.raises(ValueError, match="evidence_types must be an array"):
        summarizer.summarize(ARTICLE)


def test_parse_rejects_private_output_markers() -> None:
    summarizer, _ = _build_summarizer(
        json.dumps(
            {
                "landuse_evidence_summary": "<think>private reasoning</think> No relevant evidence.",
                "landcover_relevance": "none",
                "evidence_types": [],
                "evidence_sentences_no_place": [],
                "uncertainty": "low",
            }
        )
    )

    with pytest.raises(ValueError, match="private thinking markers"):
        summarizer.summarize(ARTICLE)


def test_parse_rejects_title_leakage_variants(fake_good_response: str) -> None:
    payload = json.loads(fake_good_response)
    payload["evidence_sentences_no_place"] = ["La source parle de La-Sainte_Marie en détail."]
    payload["landuse_evidence_summary"] = "L'article évoque La Sainte Marie."
    summarizer, _ = _build_summarizer(json.dumps(payload))

    with pytest.raises(ValueError, match="place name leakage"):
        summarizer.summarize(ARTICLE)


def test_parse_rejects_title_leakage_compact_form(fake_good_response: str) -> None:
    payload = json.loads(fake_good_response)
    payload["evidence_sentences_no_place"] = ["Les données mentionnent LaSainteMarie."]
    summarizer = LandUseEvidenceSummarizer(
        model_path="test_model.gguf",
        client=_FakeClient(response=json.dumps(payload)),
    )

    with pytest.raises(ValueError, match="place name leakage"):
        summarizer.summarize(
            {
                "title": "La-Sainte_Marie",
                "content": "Lieux sans titre.",
                "url": "https://fr.wikipedia.org/wiki/La-Sainte_Marie",
            }
        )


def test_does_not_reject_short_title_subword_match() -> None:
    payload = {
        "landuse_evidence_summary": "Le paysage montre de nombreux coteaux plantés en vignes.",
        "landcover_relevance": "medium",
        "evidence_types": ["vineyard"],
        "evidence_sentences_no_place": ["des vignes couvrent les coteaux."],
        "uncertainty": "low",
    }
    summarizer = LandUseEvidenceSummarizer(
        model_path="test_model.gguf",
        client=_FakeClient(response=json.dumps(payload)),
    )

    result = summarizer.summarize(
        {
            "title": "Vigne",
            "content": "L'article mentionne des cultures locales.",
            "url": "https://fr.wikipedia.org/wiki/Vigne",
        }
    )

    assert result["landuse_evidence_summary"] == payload["landuse_evidence_summary"]


def test_parse_allows_repeated_whitespace_and_normalizes() -> None:
    summarizer, _ = _build_summarizer(
        json.dumps(
            {
                "landuse_evidence_summary": "  This contains   many   spaces. ",
                "landcover_relevance": "low",
                "evidence_types": ["urban_or_artificial"],
                "evidence_sentences_no_place": ["  spaced\\nsentence.   ", "two  spaces"],
                "uncertainty": "medium",
            }
        )
    )

    result = summarizer.summarize(ARTICLE)

    assert result["landuse_evidence_summary"] == "This contains many spaces."
    assert result["evidence_sentences_no_place"] == ["spaced sentence.", "two spaces"]


def test_process_file_skips_current_records_regenerates_malformed_or_stale(tmp_path: Any) -> None:
    input_path = tmp_path / "articles.json"
    output_path = tmp_path / "evidence.json"
    input_path.write_text(json.dumps({"1": ARTICLE, "2": ARTICLE, "4": ARTICLE}), encoding="utf-8")

    valid_record = {
        "title": ARTICLE["title"],
        "content": ARTICLE["content"],
        "url": ARTICLE["url"],
        "landuse_evidence_summary": "L'article mentionne de l'eau et des zones urbaines.",
        "landcover_relevance": "high",
        "evidence_types": ["water", "urban_or_artificial"],
        "evidence_sentences_no_place": ["des terres agricoles sont présentes"],
        "uncertainty": "low",
        "landuse_evidence_summary_char_count": 51,
        "evidence_sentences_count": 1,
        "metadata": {
            "evidence_mode": "landuse_evidence",
            "prompt_version": LANDUSE_EVIDENCE_PROMPT_VERSION,
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
        "title": ARTICLE["title"],
        "content": ARTICLE["content"],
        "url": ARTICLE["url"],
        "landuse_evidence_summary": "invalid",
        "landcover_relevance": "unknown",
        "evidence_types": ["forest"],
        "evidence_sentences_no_place": [""],
        "uncertainty": "low",
        "landuse_evidence_summary_char_count": 7,
        "evidence_sentences_count": 1,
    }
    wrong_count_record = {
        "title": ARTICLE["title"],
        "content": ARTICLE["content"],
        "url": ARTICLE["url"],
        "landuse_evidence_summary": "La commune évoque des vignes et des zones d'eau.",
        "landcover_relevance": "low",
        "evidence_types": ["water", "vineyard"],
        "evidence_sentences_no_place": ["Des vignes sont notables sur la commune."],
        "uncertainty": "low",
        "landuse_evidence_summary_char_count": 4,
        "evidence_sentences_count": 2,
        "metadata": {
            "evidence_mode": "landuse_evidence",
            "prompt_version": LANDUSE_EVIDENCE_PROMPT_VERSION,
            "summary_no_place": True,
            "model": "Qwen3.6-27B-Q4_0.gguf",
            "model_repo_id": None,
            "seed": 42,
            "temperature": 0.0,
            "prompt": "prompt",
            "system_prompt": "system",
        },
    }
    stale_record = {
        "title": "Old",
        "content": "Old",
        "landuse_evidence_summary": "old",
        "landcover_relevance": "low",
        "evidence_types": ["forest"],
        "evidence_sentences_no_place": ["old"],
        "uncertainty": "low",
        "landuse_evidence_summary_char_count": 3,
        "evidence_sentences_count": 1,
    }
    output_path.write_text(
        json.dumps(
            {
                "1": valid_record,
                "2": malformed_record,
                "3": stale_record,
                "4": wrong_count_record,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = json.dumps(
        {
            "landuse_evidence_summary": "L'article mentionne une zone boisée et des vignes.",
            "landcover_relevance": "medium",
            "evidence_types": ["forest", "vineyard"],
            "evidence_sentences_no_place": ["Une forêt et des vignes existent."],
            "uncertainty": "low",
        }
    )
    client = _FakeClient(response=response)
    summarizer = LandUseEvidenceSummarizer(model_path=None, client=client)
    summarizer.process_file(str(input_path), str(output_path))

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(result) == {"1", "2", "4"}
    assert len(client.calls) == 2
    assert result["1"]["landuse_evidence_summary"] == valid_record["landuse_evidence_summary"]
    assert (
        result["2"]["landuse_evidence_summary"]
        == "L'article mentionne une zone boisée et des vignes."
    )
    assert (
        result["4"]["landuse_evidence_summary"]
        == "L'article mentionne une zone boisée et des vignes."
    )
    assert result["4"]["metadata"]["attempt_count"] == 1


def test_process_file_default_output_path_matches_data_paths() -> None:
    assert DataPaths().article_landuse_evidence_summaries == (
        "data/wiki/article_landuse_evidence_summaries.json"
    )
