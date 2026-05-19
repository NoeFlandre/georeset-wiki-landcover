"""Fetch Experiment 014 Sentinel-2 RGB patch caches at multiple physical scales."""

from __future__ import annotations

import argparse
from pathlib import Path

from georeset.experiment_paths import experiment_artifact_dir, experiment_artifact_file
from georeset.vision.sentinel_multiscale_patches import (
    sentinel2_planetary_computer_multiscale_fetcher,
    write_multiscale_patch_caches,
)

EXPERIMENT_ID = "quality_weighted_multiscale_image_probe_v1"


def _parse_windows(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=experiment_artifact_file(EXPERIMENT_ID, "image_probe_splits_v2.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=experiment_artifact_dir(EXPERIMENT_ID))
    parser.add_argument("--window-m", default="320,640,1280,2240")
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--cloud-cover", type=float, default=25.0)
    parser.add_argument("--datetime-range", default="2022-04-01/2022-10-31")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fetcher = sentinel2_planetary_computer_multiscale_fetcher(
        cloud_cover=args.cloud_cover,
        datetime_range=args.datetime_range,
    )
    write_multiscale_patch_caches(
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        window_m_values=_parse_windows(args.window_m),
        output_size=args.output_size,
        fetcher=fetcher,
    )


if __name__ == "__main__":
    main()
