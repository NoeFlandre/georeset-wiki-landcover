"""Summarize OSM polygons class distribution counts"""

import argparse
from typing import cast

import pandas as pd


def class_count_summary(csv_path: str) -> pd.DataFrame:
    """
    For each OSM polygon, we count how many CORINE classes intersect it.
    We return a summary table of how many polygons have n classes.
    """

    df = pd.read_csv(csv_path)

    class_counts = df.groupby("osm_id").size()

    summary = class_counts.value_counts().sort_index().reset_index()

    summary.columns = ["n_classes", "n_polygons"]
    summary["pct_polygons"] = (summary["n_polygons"] / len(class_counts) * 100).round(2)
    return summary


def format_class_count_summary(summary: pd.DataFrame) -> str:
    """Format a class-count summary table for human-readable output."""

    return cast(str, summary.to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the number of CORINE classes per OSM polygon."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/distribution/osm_corine_distribution.csv",
        help="Path to the OSM/CORINE distribution CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = class_count_summary(args.csv_path)
    print(format_class_count_summary(summary))


if __name__ == "__main__":
    main()
