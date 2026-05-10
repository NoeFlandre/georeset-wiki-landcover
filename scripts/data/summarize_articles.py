"""Summarize Wikipedia articles using an LLM."""

import argparse
import logging
import os
import sys

# Ensure src is in path if running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import DataPaths, ModelSettings
from src.fetchers.article_summarizer import ArticleSummarizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for local and Grid5000 summarization runs."""
    data_paths = DataPaths()
    model_settings = ModelSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Summarize fetched Wikipedia articles with a local GGUF model."
    )
    parser.add_argument(
        "--input-path",
        default=data_paths.article_contents,
        help="Path to fetched article contents JSON.",
    )
    parser.add_argument(
        "--output-path",
        default=data_paths.article_summaries,
        help="Path where resumable summaries JSON is written.",
    )
    parser.add_argument(
        "--model-path",
        default=model_settings.model_path,
        help="GGUF filename or path passed to llama_cpp.Llama.from_pretrained.",
    )
    parser.add_argument(
        "--seed", type=int, default=model_settings.seed, help="Deterministic generation seed."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=model_settings.summarization_temperature,
        help="Sampling temperature.",
    )
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
