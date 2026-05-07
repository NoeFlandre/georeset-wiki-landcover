"""Summarize Wikipedia articles using an LLM."""

import argparse
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class ArticleSummarizer:
    """Summarizes Wikipedia articles using a local LLM via llama-cpp-python."""

    SUMMARY_SCHEMA = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A concise one-sentence French summary of the Wikipedia article"
            }
        },
        "required": ["summary"],
        "additionalProperties": False
    }
    PRIVATE_OUTPUT_MARKERS = (
        "<think",
        "</think>",
        "here's a thinking process",
        "output matches",
    )

    def __init__(self, model_path: str, seed: int = 42, temperature: float = 0.7):
        self.model_path = model_path
        self.seed = seed
        self.temperature = temperature
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
                n_ctx=8192
            )
        return self._llm

    def summarize(self, article: Dict) -> Dict:
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
        prompt = f"Résumez cet article Wikipedia en une phrase concise:\n\n{content}"
        result["summary"] = self._generate_summary(prompt)

        result["metadata"] = {
            "model": self.model_path,
            "seed": self.seed,
            "temperature": self.temperature,
            "prompt": prompt
        }
        return result

    def _generate_summary(self, prompt: str) -> str:
        """Call the LLM with structured JSON output."""
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant qui résume des articles Wikipedia. "
                        "Réponds uniquement avec un objet JSON qui respecte le schéma fourni. "
                        "N'inclus aucun raisonnement, balise <think>, Markdown ou champ supplémentaire."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=256,
            response_format={
                "type": "json_object",
                "schema": self.SUMMARY_SCHEMA
            }
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

        unexpected_keys = set(payload) - set(self.SUMMARY_SCHEMA["properties"])
        if unexpected_keys:
            raise ValueError(f"LLM summary JSON contains unexpected keys: {sorted(unexpected_keys)}")

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

        to_process = {
            k: v for k, v in articles.items()
            if k not in existing or "summary" not in existing[k]
        }

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for local and Grid5000 summarization runs."""
    parser = argparse.ArgumentParser(description="Summarize fetched Wikipedia articles with a local GGUF model.")
    parser.add_argument(
        "--input-path",
        default="data/wiki/article_contents.json",
        help="Path to fetched article contents JSON.",
    )
    parser.add_argument(
        "--output-path",
        default="data/wiki/article_summaries.json",
        help="Path where resumable summaries JSON is written.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("GEORESET_MODEL_PATH", "Qwen3.6-27B-Q4_0.gguf"),
        help="GGUF filename or path passed to llama_cpp.Llama.from_pretrained.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generation seed.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summarizer = ArticleSummarizer(
        model_path=args.model_path,
        seed=args.seed,
        temperature=args.temperature,
    )
    summarizer.process_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
