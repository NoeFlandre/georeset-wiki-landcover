"""Fetch Sentinel-2 RGB patches for CLIP weak-label examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from georeset.vision.sentinel_patches import (
    sentinel2_planetary_computer_fetcher,
    write_patch_cache,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=Path("data/experiments/clip_linear_probe_weak_labels_v1/label_splits.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/experiments/clip_linear_probe_weak_labels_v1/sentinel_patches_rgb.npz"),
    )
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--cloud-cover", type=float, default=25.0)
    parser.add_argument("--datetime-range", default="2022-04-01/2022-10-31")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fetcher = sentinel2_planetary_computer_fetcher(
        patch_size=args.patch_size,
        cloud_cover=args.cloud_cover,
        datetime_range=args.datetime_range,
    )
    write_patch_cache(splits_path=args.splits_path, output_path=args.output_path, fetcher=fetcher)


if __name__ == "__main__":
    main()
