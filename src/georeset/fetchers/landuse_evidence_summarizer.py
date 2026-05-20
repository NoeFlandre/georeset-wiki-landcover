"""Summarizer that extracts land-use evidence from Wikipedia article text."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, cast

from georeset.config import DataPaths
from georeset.contracts import ArticleContent, SummaryRecord
from georeset.llm.llama_client import DEFAULT_GGUF_FILENAME, JsonChatClient, LlamaChatClient
from georeset.utils.json_io import read_json_file, write_json_atomic

logger = logging.getLogger(__name__)

LANDUSE_EVIDENCE_PROMPT_VERSION = 2
EVIDENCE_TYPES = (
    "forest",
    "agriculture",
    "vineyard",
    "pasture",
    "water",
    "wetland",
    "shrubland",
    "bare_ground",
    "urban_or_artificial",
    "relief_or_geology",
    "habitat_or_ecology",
)


class LandUseEvidenceSummarizer:
    """Extract a no-place land-use evidence summary from one Wikipedia article."""

    EVIDENCE_SCHEMA = {
        "type": "object",
        "properties": {
            "landuse_evidence_summary": {
                "type": "string",
                "description": "Concise no-place land-use summary in French",
            },
            "landcover_relevance": {
                "type": "string",
                "enum": ["none", "low", "medium", "high"],
            },
            "evidence_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(EVIDENCE_TYPES)},
            },
            "evidence_sentences_no_place": {
                "type": "array",
                "items": {"type": "string"},
            },
            "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": [
            "landuse_evidence_summary",
            "landcover_relevance",
            "evidence_types",
            "evidence_sentences_no_place",
            "uncertainty",
        ],
        "additionalProperties": False,
    }
    PRIVATE_OUTPUT_MARKERS = (
        "<think",
        "</think>",
        "here's a thinking process",
        "output matches",
    )
    LANDCOVER_RELEVANCE = ("none", "low", "medium", "high")
    UNCERTAINTY_LEVELS = ("low", "medium", "high")

    def __init__(
        self,
        model_path: str | None = None,
        model_repo_id: str | None = None,
        seed: int = 42,
        temperature: float = 0.0,
        max_attempts: int = 2,
        client: JsonChatClient | None = None,
    ) -> None:
        self.model_path = model_path or DEFAULT_GGUF_FILENAME
        self.model_repo_id = model_repo_id
        self.seed = seed
        self.temperature = temperature
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self._client = client or LlamaChatClient(
            model_path=self.model_path,
            seed=seed,
            repo_id=model_repo_id,
        )

    @staticmethod
    def _normalize_for_whitespace(value: str) -> str:
        normalized = value.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        return " ".join(normalized.split())

    @staticmethod
    def _normalize_for_leak(value: str) -> str:
        normalized = value.lower()
        normalized = normalized.replace("’", "'")
        normalized = normalized.replace("_", " ")
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[\"'‘’`~^*+<>.,;:!?/\\(){}\[\]]", " ", normalized)
        normalized = " ".join(normalized.split())
        return normalized

    @classmethod
    def _title_variants(cls, title: str) -> set[str]:
        if not title:
            return set()
        normalized = cls._normalize_for_leak(title)
        compact = normalized.replace(" ", "")
        return {
            title.lower().strip(),
            normalized,
            compact,
        } - {""}

    @staticmethod
    def _has_word_boundary_match(text: str, pattern: str) -> bool:
        escaped = re.escape(pattern)
        boundary_pattern = rf"(?<!\w){escaped}(?!\w)"
        return bool(re.search(boundary_pattern, text))

    @classmethod
    def _contains_title_leak(cls, text: str, title: str) -> bool:
        if not title:
            return False
        normalized_text = cls._normalize_for_leak(text).lower()
        compact_text = normalized_text.replace(" ", "")
        for variant in cls._title_variants(title):
            normalized_variant = cls._normalize_for_leak(variant).lower()
            if not normalized_variant:
                continue
            if cls._has_word_boundary_match(normalized_text, normalized_variant):
                return True

            compact_variant = normalized_variant.replace(" ", "")
            if len(compact_variant) >= 8 and compact_variant in compact_text:
                return True
        return False

    def summarize(self, article: ArticleContent) -> SummaryRecord:
        """Add land-use evidence fields to an article dict."""
        result = cast(SummaryRecord, dict(article))

        max_content_chars = 24000
        content = article["content"]
        if len(content) > max_content_chars:
            content = content[:max_content_chars]

        user_prompt = self._user_prompt(article, content)
        system_prompt = (
            "Tu es un assistant d'extraction d'indices d'occupation du sol à partir "
            "d'articles Wikipédia en français.\n\n"
            "Ta tâche n'est pas de résumer l'article en général, mais d'extraire uniquement "
            "les informations utiles pour décrire l'occupation du sol, l'usage foncier, "
            "le paysage, la végétation, l'agriculture, l'eau, les zones humides, le relief, "
            "la géologie, les habitats ou l'écologie.\n\n"
            "Réponds uniquement avec un objet JSON valide respectant le schéma fourni. "
            "N'ajoute aucun Markdown, aucune explication, aucun champ supplémentaire."
        )
        last_error: ValueError | None = None
        payload: dict[str, Any]
        successful_user_prompt = user_prompt
        attempt_count = 0
        for attempt in range(1, self.max_attempts + 1):
            attempt_count = attempt
            current_user_prompt = (
                self._retry_user_prompt(user_prompt) if attempt > 1 else user_prompt
            )
            try:
                successful_user_prompt = current_user_prompt
                payload = self._parse_payload(
                    self._generate_landuse_evidence(current_user_prompt, system_prompt)
                )
                payload = self._validate_no_place(payload, article)
                break
            except ValueError as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    raise
        else:
            raise ValueError("Failed to parse valid land-use evidence JSON") from last_error

        result["landuse_evidence_summary"] = payload["landuse_evidence_summary"]
        result["landcover_relevance"] = payload["landcover_relevance"]
        result["evidence_types"] = payload["evidence_types"]
        result["evidence_sentences_no_place"] = payload["evidence_sentences_no_place"]
        result["uncertainty"] = payload["uncertainty"]
        result["landuse_evidence_summary_char_count"] = len(payload["landuse_evidence_summary"])
        result["evidence_sentences_count"] = len(payload["evidence_sentences_no_place"])

        result["metadata"] = {
            "model": self.model_path,
            "model_repo_id": self.model_repo_id,
            "seed": self.seed,
            "temperature": self.temperature,
            "evidence_mode": "landuse_evidence",
            "prompt_version": LANDUSE_EVIDENCE_PROMPT_VERSION,
            "summary_no_place": True,
            "prompt": successful_user_prompt,
            "system_prompt": system_prompt,
            "attempt_count": attempt_count,
        }
        return result

    def _generate_landuse_evidence(self, user_prompt: str, system_prompt: str) -> str:
        return self._client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=self.EVIDENCE_SCHEMA,
            temperature=self.temperature,
            max_tokens=512,
        )

    def _parse_payload(self, raw_content: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM did not return valid land-use evidence JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("LLM land-use evidence JSON must be an object")

        payload_dict = cast(dict[str, Any], payload)
        required_fields = set(cast(list[str], self.EVIDENCE_SCHEMA["required"]))
        unexpected_fields = set(cast(dict[str, Any], self.EVIDENCE_SCHEMA["properties"]))
        missing = required_fields - set(payload_dict)
        if missing:
            raise ValueError(
                f"LLM land-use evidence JSON missing required fields: {sorted(missing)}"
            )

        unexpected = set(payload_dict) - unexpected_fields
        if unexpected:
            raise ValueError(
                f"LLM land-use evidence JSON has unexpected fields: {sorted(unexpected)}"
            )

        summary = payload_dict["landuse_evidence_summary"]
        landcover_relevance = payload_dict["landcover_relevance"]
        evidence_types = payload_dict["evidence_types"]
        evidence_sentences = payload_dict["evidence_sentences_no_place"]
        uncertainty = payload_dict["uncertainty"]

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("LLM land-use evidence JSON must include a non-empty summary")
        summary = self._normalize_for_whitespace(summary)

        if any(marker in summary.lower() for marker in self.PRIVATE_OUTPUT_MARKERS):
            raise ValueError("LLM land-use evidence response contains private thinking markers")

        if (
            not isinstance(landcover_relevance, str)
            or landcover_relevance not in self.LANDCOVER_RELEVANCE
        ):
            raise ValueError("LLM land-use evidence JSON has invalid landcover_relevance")

        if not isinstance(uncertainty, str) or uncertainty not in self.UNCERTAINTY_LEVELS:
            raise ValueError("LLM land-use evidence JSON has invalid uncertainty")

        if not isinstance(evidence_types, list):
            raise ValueError("LLM land-use evidence JSON field evidence_types must be an array")

        invalid_evidence = [
            item
            for item in evidence_types
            if not isinstance(item, str) or item not in EVIDENCE_TYPES
        ]
        if invalid_evidence:
            raise ValueError(
                "LLM land-use evidence JSON has invalid evidence_types values: "
                f"{sorted(set(invalid_evidence))}"
            )

        if not isinstance(evidence_sentences, list):
            raise ValueError(
                "LLM land-use evidence JSON field evidence_sentences_no_place must be an array"
            )
        normalized_evidence: list[str] = []
        for sentence in evidence_sentences:
            if not isinstance(sentence, str):
                raise ValueError(
                    "LLM land-use evidence JSON field evidence_sentences_no_place must be strings"
                )
            normalized_sentence = self._normalize_for_whitespace(sentence)
            if any(marker in normalized_sentence.lower() for marker in self.PRIVATE_OUTPUT_MARKERS):
                raise ValueError("LLM land-use evidence response contains private thinking markers")
            normalized_evidence.append(normalized_sentence)

        if landcover_relevance == "none":
            if evidence_types:
                raise ValueError(
                    "LLM land-use evidence JSON must omit evidence_types when relevance is none"
                )
            if evidence_sentences:
                raise ValueError(
                    "LLM land-use evidence JSON must omit evidence_sentences_no_place when relevance is none"
                )
            if len(summary) > 240:
                raise ValueError(
                    "LLM land-use evidence JSON summary for no evidence should be short"
                )

        return {
            "landuse_evidence_summary": summary,
            "landcover_relevance": landcover_relevance,
            "evidence_types": [str(item) for item in evidence_types],
            "evidence_sentences_no_place": normalized_evidence,
            "uncertainty": uncertainty,
        }

    def _validate_no_place(
        self, payload: dict[str, Any], article: ArticleContent
    ) -> dict[str, Any]:
        title = str(article.get("title", ""))

        if title and self._contains_title_leak(payload["landuse_evidence_summary"], title):
            raise ValueError("LLM land-use evidence response contains place name leakage")

        for sentence in payload["evidence_sentences_no_place"]:
            if title and self._contains_title_leak(sentence, title):
                raise ValueError("LLM land-use evidence response contains place name leakage")

        return payload

    @staticmethod
    def _remove_private_fields(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: LandUseEvidenceSummarizer._remove_private_fields(item)
                for key, item in value.items()
                if key != "thinking"
            }
        if isinstance(value, list):
            return [LandUseEvidenceSummarizer._remove_private_fields(item) for item in value]
        return value

    @staticmethod
    def _user_prompt(article: ArticleContent, content: str) -> str:
        title = str(article.get("title", ""))
        title_forbidden = (
            f"Titre de l'article à ne jamais mentionner ni utiliser comme indice:\n{title}\n\n"
            if title
            else ""
        )
        return (
            f"{title_forbidden}"
            f"Texte de l'article:\n{content}\n\n"
            "Extrais les indices utiles pour l'occupation du sol et l'usage foncier.\n\n"
            "Contraintes:\n"
            "- N'utilise pas le titre comme preuve.\n"
            "- Ne mentionne jamais le titre, le nom du lieu décrit, ses variantes évidentes, "
            "ni des coordonnées.\n"
            "- Ignore les informations non pertinentes: histoire, dates, monuments, bâtiments, "
            "population, administration, tourisme, transport, biographies, culture, politique, "
            "étymologie.\n"
            "- Ne déduis rien qui n'est pas explicitement présent dans le texte.\n"
            "- Reformule les preuves en français clair, sans copier de longues phrases "
            "textuellement.\n\n"
            "Types d'indices autorisés pour evidence_types:\n"
            "forest, agriculture, vineyard, pasture, water, "
            "wetland, shrubland, bare_ground, urban_or_artificial, relief_or_geology, "
            "habitat_or_ecology.\n\n"
            "Définitions de landcover_relevance:\n"
            "- none: aucune information utile sur l'occupation du sol ou le paysage.\n"
            "- low: indice faible, indirect ou très général.\n"
            "- medium: plusieurs indices utiles mais incomplets.\n"
            "- high: indices clairs et directement utiles pour classifier le paysage ou "
            "l'usage du sol.\n\n"
            "Retourne exactement cet objet JSON:\n"
            "{\n"
            '  "landuse_evidence_summary": "1 à 3 phrases courtes, sans nom de lieu, '
            "résumant uniquement les indices utiles d'occupation du sol.\",\n"
            '  "landcover_relevance": "none|low|medium|high",\n'
            '  "evidence_types": ["liste de types parmi les types autorisés"],\n'
            '  "evidence_sentences_no_place": ["phrases factuelles reformulées, sans nom de lieu"],\n'
            '  "uncertainty": "low|medium|high"\n'
            "}\n\n"
            "Si aucune preuve utile n'est présente, retourne:\n"
            "{\n"
            '  "landuse_evidence_summary": "Aucune preuve utile sur l\'occupation du sol ou '
            "le paysage n'est présente dans le texte.\",\n"
            '  "landcover_relevance": "none",\n'
            '  "evidence_types": [],\n'
            '  "evidence_sentences_no_place": [],\n'
            '  "uncertainty": "high"\n'
            "}"
        )

    @staticmethod
    def _retry_user_prompt(user_prompt: str) -> str:
        return (
            f"{user_prompt}\n\n"
            "Correction: ta réponse doit être uniquement un JSON valide conforme au schéma "
            "ci-dessus (clés attendues uniquement), sans Markdown ni explication, et ne doit "
            "en aucun cas réintroduire le nom du lieu/article. Réponds strictement en JSON."
        )

    @staticmethod
    def _extract_payload_fields(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "landuse_evidence_summary": record["landuse_evidence_summary"],
            "landcover_relevance": record["landcover_relevance"],
            "evidence_types": record["evidence_types"],
            "evidence_sentences_no_place": record["evidence_sentences_no_place"],
            "uncertainty": record["uncertainty"],
        }

    def _has_current_record(
        self,
        article: ArticleContent,
        record: dict[str, Any] | None,
    ) -> bool:
        if not isinstance(record, dict):
            return False
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            return False

        expected_metadata = {
            "evidence_mode": "landuse_evidence",
            "prompt_version": LANDUSE_EVIDENCE_PROMPT_VERSION,
            "summary_no_place": True,
            "model": self.model_path,
            "model_repo_id": self.model_repo_id,
            "seed": self.seed,
            "temperature": self.temperature,
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                return False

        required_fields = {
            "landuse_evidence_summary",
            "landcover_relevance",
            "evidence_types",
            "evidence_sentences_no_place",
            "uncertainty",
            "landuse_evidence_summary_char_count",
            "evidence_sentences_count",
        }
        if not required_fields.issubset(record):
            return False

        try:
            parsed: dict[str, Any] = self._parse_payload(
                json.dumps(self._extract_payload_fields(record))
            )
            parsed = self._validate_no_place(parsed, article)
        except ValueError:
            return False

        normalized_summary = self._normalize_for_whitespace(record["landuse_evidence_summary"])
        if parsed["landuse_evidence_summary"] != normalized_summary:
            return False
        if (
            not isinstance(record["landuse_evidence_summary_char_count"], int | float | str)
            or str(record["landuse_evidence_summary_char_count"]).strip() == ""
        ):
            return False
        try:
            if len(normalized_summary) != int(record["landuse_evidence_summary_char_count"]):
                return False
        except (TypeError, ValueError):
            return False

        normalized_sentences = [
            self._normalize_for_whitespace(sentence)
            for sentence in cast(list[str], record["evidence_sentences_no_place"])
        ]
        if (
            not isinstance(record["evidence_sentences_count"], int | float | str)
            or str(record["evidence_sentences_count"]).strip() == ""
        ):
            return False
        try:
            if len(normalized_sentences) != int(record["evidence_sentences_count"]):
                return False
        except (TypeError, ValueError):
            return False
        return cast(list[str], parsed["evidence_sentences_no_place"]) == normalized_sentences

    def process_file(self, input_path: str, output_path: str) -> None:
        """Summarize all input articles and write the output resume file.

        Reads existing output, prunes stale page IDs, and reprocesses malformed or stale records.
        """

        raw_existing: object = {}
        if os.path.exists(output_path):
            raw_existing = read_json_file(output_path)

        existing: dict[str, SummaryRecord] = {}
        if isinstance(raw_existing, dict):
            existing = cast(dict[str, SummaryRecord], self._remove_private_fields(raw_existing))

        articles = cast(dict[str, ArticleContent], read_json_file(input_path))
        valid_keys = {str(k) for k in articles}
        existing = {k: v for k, v in existing.items() if str(k) in valid_keys}

        to_process = {
            k: article
            for k, article in articles.items()
            if not self._has_current_record(article, cast(dict[str, Any] | None, existing.get(k)))
        }

        logger.info("Processing %s of %s articles...", len(to_process), len(articles))

        failed_summaries: list[tuple[str, str]] = []

        for i, (pageid, article) in enumerate(to_process.items(), 1):
            logger.info(
                "[%s/%s] Extracting land-use evidence: %s",
                i,
                len(to_process),
                article.get("title", pageid),
            )
            try:
                existing[pageid] = self.summarize(article)
            except ValueError as exc:
                failed_summarization = (pageid, str(article.get("title", "")))
                failed_summaries.append(failed_summarization)
                existing.pop(pageid, None)
                logger.error(
                    "Failed to summarize land-use evidence for pageid=%s title=%s: %s",
                    pageid,
                    article.get("title", ""),
                    exc,
                )
                continue
            write_json_atomic(output_path, existing, indent=2, ensure_ascii=False)

        write_json_atomic(output_path, existing, indent=2, ensure_ascii=False)
        if failed_summaries:
            failed_count = len(failed_summaries)
            logger.warning(
                "Land-use evidence summarization completed with %s failed articles.",
                failed_count,
            )

        logger.info("Done. Saved %s summaries to %s", len(existing), output_path)


if __name__ == "__main__":
    paths = DataPaths()
    summarizer = LandUseEvidenceSummarizer(model_path="Qwen3.6-27B-Q4_0.gguf")
    summarizer.process_file(paths.article_contents, paths.article_landuse_evidence_summaries)
