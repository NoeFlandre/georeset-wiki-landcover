"""Summarize Wikipedia articles using an LLM."""

import json
import os
from typing import Dict


class ArticleSummarizer:
    """Summarizes Wikipedia articles using a local LLM via llama-cpp-python."""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to the GGUF model file.
        """
        self.model_path = model_path
        self._llm = None

    def _get_llm(self):
        """Lazy initialization of LLM (GPU-accelerated)."""
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama.from_pretrained(
                repo_id="unsloth/Qwen3.6-27B-GGUF",
                filename="Qwen3.6-27B-Q4_0.gguf",
            )
        return self._llm

    def summarize(self, article: Dict) -> Dict:
        """
        Add a summary to an article dict.

        Args:
            article: Dict with title, content, url keys.

        Returns:
            New dict with all original keys plus 'summary'.
        """
        result = dict(article)
        result["summary"] = self._call_llm(article["content"])
        return result

    def _call_llm(self, content: str) -> str:
        """Call the LLM to summarize content."""
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"Résumez cet article Wikipedia en une phrase concise:\n\n{content}"
                }
            ]
        )
        return response["choices"][0]["message"]["content"]

    def process_file(self, input_path: str, output_path: str):
        """
        Process all articles from input_path and save summaries to output_path.

        Skips articles that already have a 'summary' key.
        Saves progress incrementally.

        Args:
            input_path: Path to article_contents.json
            output_path: Path to save summarized articles
        """
        # Load existing output (for resume)
        existing = {}
        if os.path.exists(output_path):
            with open(output_path) as f:
                existing = json.load(f)

        # Load input articles
        with open(input_path) as f:
            articles = json.load(f)

        # Process only articles missing summary
        to_process = {
            k: v for k, v in articles.items()
            if k not in existing or "summary" not in existing[k]
        }

        print(f"Processing {len(to_process)} of {len(articles)} articles...")

        for i, (pageid, article) in enumerate(to_process.items(), 1):
            print(f"[{i}/{len(to_process)}] Summarizing: {article.get('title', pageid)}")
            result = self.summarize(article)
            existing[pageid] = result

            # Incremental save every 50 articles
            if i % 50 == 0:
                with open(output_path, "w") as f:
                    json.dump(existing, f, indent=2)
                print(f"  Checkpoint saved ({i} processed)")

        # Final save
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"Done. Saved {len(existing)} summaries to {output_path}")


if __name__ == "__main__":
    summarizer = ArticleSummarizer(model_path="Qwen3.6-27B-Q4_0.gguf")
    summarizer.process_file(
        "data/wiki/article_contents.json",
        "data/wiki/article_summaries.json"
    )
