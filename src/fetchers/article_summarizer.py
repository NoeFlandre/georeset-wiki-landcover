"""Logic for summarizing Wikipedia articles using a local LLM."""

import json
import logging
import os
from typing import Any, cast

logger = logging.getLogger(__name__)


class ArticleSummarizer:
    """Summarizes Wikipedia articles using a local LLM via llama-cpp-python."""

    SUMMARY_SCHEMA = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise one-sentence French summary of the Wikipedia article",
            }
        },
        "required": ["summary"],
        "additionalProperties": False,
    }
    PRIVATE_OUTPUT_MARKERS = (
        "<think",
        "</think>",
        "here's a thinking process",
        "output matches",
    )
    SUMMARY_MODES = ("place", "no_place")

    def __init__(
        self,
        model_path: str,
        seed: int = 42,
        temperature: float = 0.7,
        summary_mode: str = "place",
    ):
        if summary_mode not in self.SUMMARY_MODES:
            raise ValueError(f"summary_mode must be one of {self.SUMMARY_MODES}")
        self.model_path = model_path
        self.seed = seed
        self.temperature = temperature
        self.summary_mode = summary_mode
        self._llm = None

    def _get_llm(self):
        """Lazy initialization of LLM (GPU-accelerated)."""
        if self._llm is None:
            import llama_cpp

            self._llm = llama_cpp.Llama.from_pretrained(
                repo_id="unsloth/Qwen3.6-27B-GGUF",
                filename=self.model_path if self.model_path else "Qwen3.6-27B-Q4_0.gguf",
                n_gpu_layers=-1,
                seed=self.seed,
                n_ctx=8192,
            )
        return self._llm

    def summarize(self, article: dict) -> dict:
        """
        Add a summary to an article dict.

        Args:
            article: Dict with title, content, url keys.

        Returns:
            New dict with all original keys plus 'summary' and 'metadata'.
        """
        result = dict(article)
        max_content_chars = 24000
        content = article["content"]
        if len(content) > max_content_chars:
            content = content[:max_content_chars]
        if self.summary_mode == "no_place":
            prompt = (
                "Résumez cet article Wikipedia en une phrase concise, sans jamais mentionner "
                f"le nom du lieu décrit:\n\n{content}"
            )
        else:
            prompt = f"Résumez cet article Wikipedia en une phrase concise:\n\n{content}"
        system_prompt = (
            "Tu es un assistant qui résume des articles Wikipedia. "
            "Réponds uniquement avec un objet JSON qui respecte le schéma fourni. "
            "N'inclus aucun raisonnement, balise <think>, Markdown ou champ supplémentaire."
        )
        result["summary"] = self._generate_summary(prompt, system_prompt)

        result["metadata"] = {
            "model": self.model_path,
            "seed": self.seed,
            "temperature": self.temperature,
            "summary_mode": self.summary_mode,
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        return result

    def _generate_summary(self, prompt: str, system_prompt: str) -> str:
        """Call the LLM with structured JSON output."""
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=256,
            response_format={"type": "json_object", "schema": self.SUMMARY_SCHEMA},
        )
        raw_content = response["choices"][0]["message"]["content"]
        return self._summary_from_response(raw_content)

    def _summary_from_response(self, raw_content: str) -> str:
        """Validate the schema-constrained model response and return the public summary."""
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM did not return valid summary JSON") from exc

        if not isinstance(payload, dict):
            raise ValueError("LLM summary JSON must be an object")

        payload_dict = cast(dict[str, Any], payload)
        unexpected_keys = set(payload_dict.keys()) - set(self.SUMMARY_SCHEMA["properties"])  # type: ignore[call-overload]
        if unexpected_keys:
            raise ValueError(
                f"LLM summary JSON contains unexpected keys: {sorted(unexpected_keys)}"
            )

        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("LLM summary JSON must contain a non-empty summary string")

        summary = summary.strip()
        lowered_summary = summary.lower()
        if any(marker in lowered_summary for marker in self.PRIVATE_OUTPUT_MARKERS):
            raise ValueError("LLM summary contains private thinking markers")

        return summary

    def _remove_private_fields(self, value: Any) -> Any:
        """Drop fields that should never be persisted in public summary output."""
        if isinstance(value, dict):
            return {
                key: self._remove_private_fields(item)
                for key, item in value.items()
                if key != "thinking"
            }
        if isinstance(value, list):
            return [self._remove_private_fields(item) for item in value]
        return value

    def _has_current_summary(self, article: dict[str, Any]) -> bool:
        if "summary" not in article:
            return False
        metadata = article.get("metadata")
        if not isinstance(metadata, dict):
            return False
        summary_mode = metadata.get("summary_mode")
        if summary_mode == self.summary_mode:
            return True

        # Legacy artifacts did not store summary_mode, but did persist the prompt.
        prompt = metadata.get("prompt")
        if not isinstance(prompt, str):
            return False
        is_no_place_prompt = "sans jamais mentionner le nom du lieu décrit" in prompt
        return (self.summary_mode == "no_place") == is_no_place_prompt

    def process_file(self, input_path: str, output_path: str):
        """
        Process all articles from input_path and save summaries to output_path.

        Skips articles that already have a 'summary' key.
        Saves progress incrementally.
        """
        existing = {}
        if os.path.exists(output_path):
            with open(output_path) as f:
                existing = self._remove_private_fields(json.load(f))

        with open(input_path) as f:
            articles = json.load(f)

        valid_keys = {str(k) for k in articles}
        existing = {k: v for k, v in existing.items() if str(k) in valid_keys}

        to_process = {k: v for k, v in articles.items() if not self._has_current_summary(existing.get(k, {}))}

        logger.info(f"Processing {len(to_process)} of {len(articles)} articles...")

        if existing and not to_process:
            with open(output_path, "w") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        for i, (pageid, article) in enumerate(to_process.items(), 1):
            logger.info(f"[{i}/{len(to_process)}] Summarizing: {article.get('title', pageid)}")
            result = self.summarize(article)
            existing[pageid] = result

            with open(output_path, "w") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            logger.info(f"  Checkpoint saved ({i} processed)")

        logger.info(f"Done. Saved {len(existing)} summaries to {output_path}")
