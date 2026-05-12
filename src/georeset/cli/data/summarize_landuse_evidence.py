"""Summarize Wikipedia articles into no-place land-use evidence JSON summaries."""

import argparse
import logging

from georeset.config import DataPaths, ModelSettings
from georeset.fetchers.landuse_evidence_summarizer import LandUseEvidenceSummarizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for land-use evidence extraction."""
    data_paths = DataPaths()
    model_settings = ModelSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Summarize land-use evidence from fetched Wikipedia articles."
    )
    parser.add_argument(
        "--input-path",
        default=data_paths.article_contents,
        help="Path to fetched article contents JSON.",
    )
    parser.add_argument(
        "--output-path",
        default=data_paths.article_landuse_evidence_summaries,
        help="Path where resumable evidence summaries JSON is written.",
    )
    parser.add_argument(
        "--model-path",
        default=model_settings.model_path,
        help="GGUF filename or path passed to llama_cpp.Llama.from_pretrained.",
    )
    parser.add_argument(
        "--model-repo-id",
        default=model_settings.model_repo_id,
        help="Optional Hugging Face repo_id for llama_cpp.Llama.from_pretrained.",
    )
    parser.add_argument("--seed", type=int, default=model_settings.seed, help="Deterministic seed")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (fixed default 0.0 for land-use evidence).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summarizer = LandUseEvidenceSummarizer(
        model_path=args.model_path,
        model_repo_id=args.model_repo_id,
        seed=args.seed,
        temperature=args.temperature,
    )
    summarizer.process_file(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
