"""Quick snapshot of the Corine Land Cover dataset for Alsace."""

import argparse

from georeset.fetchers.data_fetcher import DataFetcher


def snapshot(n_samples: int = 10) -> None:
    fetcher = DataFetcher()
    gdf = fetcher.load_data(exclude_artificial=True)

    print("=== Dataset Snapshot ===")
    print(f"Total polygons: {len(gdf)}")
    print(f"Bounds: {fetcher.get_bounds()}")
    print(f"Columns: {list(gdf.columns)}")

    # Class distribution at level 1
    gdf["level1"] = gdf["code_18"].str[:1]
    print("\n--- Level 1 Class Distribution ---")
    print(gdf["level1"].value_counts())

    # Distribution of polygon counts by number of classes (based on unique code_18 values per polygon)
    class_counts = gdf.groupby("ID")["code_18"].nunique()
    print("\n--- Polygons by Number of Unique Classes ---")
    print(class_counts.value_counts().sort_index())

    # Sample polygons
    print("\n--- Sample Polygons ---")
    sample = fetcher.get_sample_polygons(n=n_samples, level=2, exclude_artificial=True)
    for _, row in sample.iterrows():
        print(
            f"  Class: {row['class_label']}, Code: {row['code_18']}, Centroid: ({row['centroid'].y:.4f}, {row['centroid'].x:.4f})"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    snapshot(n_samples=args.n_samples)


if __name__ == "__main__":
    main()
