"""Quick snapshot of the Corine Land Cover dataset for Alsace."""

import argparse

from georeset.fetchers.data_fetcher import DataFetcher


def snapshot(n_samples: int = 10) -> str:
    fetcher = DataFetcher()
    gdf = fetcher.load_data(exclude_artificial=True)

    report_parts: list[str] = []
    report_parts.extend(
        [
            "=== Dataset Snapshot ===",
            f"Total polygons: {len(gdf)}",
            f"Bounds: {fetcher.get_bounds()}",
            f"Columns: {list(gdf.columns)}",
            "",
            "--- Level 1 Class Distribution ---",
            f"{gdf['code_18'].str[:1].value_counts()}",
        ]
    )

    # Distribution of polygon counts by number of classes (based on unique code_18 values per polygon)
    class_counts = gdf.groupby("ID")["code_18"].nunique()
    report_parts.extend(
        [
            "",
            "--- Polygons by Number of Unique Classes ---",
            f"{class_counts.value_counts().sort_index()}",
            "",
            "--- Sample Polygons ---",
        ]
    )
    # Sample polygons
    sample = fetcher.get_sample_polygons(n=n_samples, level=2, exclude_artificial=True)
    for _, row in sample.iterrows():
        report_parts.append(
            f"  Class: {row['class_label']}, Code: {row['code_18']}, Centroid: ({row['centroid'].y:.4f}, {row['centroid'].x:.4f})"
        )
    return "\n".join(report_parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(snapshot(n_samples=args.n_samples))


if __name__ == "__main__":
    main()
