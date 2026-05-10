"""Quick snapshot of the Corine Land Cover dataset for Alsace."""

import os
import sys

# Ensure src is in path if running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.fetchers.data_fetcher import DataFetcher


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


if __name__ == "__main__":
    snapshot()
