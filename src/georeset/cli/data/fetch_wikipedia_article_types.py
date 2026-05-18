"""CLI entrypoint for fetching Wikipedia article-type metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from georeset.experiment_paths import experiment_artifact_file
from georeset.fetchers.wiki_article_type_fetcher import WikiArticleTypeFetcher

DEFAULT_INPUT_PATH = Path("data/wiki/wiki_articles.json")
DEFAULT_OUTPUT_PATH = experiment_artifact_file(
    "article_text_classification_article_type_relevance_stratified_v1",
    "article_type_metadata.json",
)
DEFAULT_BATCH_SIZE = 50


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to wiki_articles.json list.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write article-type metadata JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of pageids per request batch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fetcher = WikiArticleTypeFetcher()
    fetcher.fetch_from_file(args.input_path, args.output_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
