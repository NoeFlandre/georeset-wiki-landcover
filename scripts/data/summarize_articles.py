"""Summarize Wikipedia articles using an LLM."""

import argparse
import logging
import os
import sys

# Ensure src is in path if running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.fetchers.article_summarizer import ArticleSummarizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for local and Grid5000 summarization runs."""
    parser = argparse.ArgumentParser(
        description="Summarize fetched Wikipedia articles with a local GGUF model."
    )
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
    parser.add_argument(
        "--summary-mode",
        choices=["place", "no_place"],
        default="place",
        help="Use place to allow place names, or no_place to suppress the described place name.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summarizer = ArticleSummarizer(
        model_path=args.model_path,
        seed=args.seed,
        temperature=args.temperature,
        summary_mode=args.summary_mode,
    )
    summarizer.process_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
