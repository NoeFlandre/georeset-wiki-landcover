from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from georeset.vision.sentinel_patches import sentinel2_planetary_computer_fetcher, write_patch_cache


def test_write_patch_cache_stores_only_successful_uint8_rgb_patches(tmp_path: Path) -> None:
    splits_path = tmp_path / "splits.csv"
    output_path = tmp_path / "patches.npz"
    pd.DataFrame(
        [
            {"pageid": "1", "lat": 48.0, "lon": 7.0},
            {"pageid": "2", "lat": 49.0, "lon": 8.0},
        ]
    ).to_csv(splits_path, index=False)

    def fetcher(row: pd.Series) -> np.ndarray | None:
        if row["pageid"] == "2":
            return None
        return np.ones((4, 4, 3), dtype=np.uint8) * 7

    write_patch_cache(splits_path=splits_path, output_path=output_path, fetcher=fetcher)

    cache = np.load(output_path)
    assert cache["pageids"].tolist() == ["1"]
    assert cache["patches"].shape == (1, 4, 4, 3)
    assert cache["patches"].dtype == np.uint8


def test_sentinel2_planetary_computer_fetcher_rejects_non_positive_patch_size() -> None:
    with pytest.raises(ValueError, match="patch_size must be positive"):
        sentinel2_planetary_computer_fetcher(
            patch_size=0,
            cloud_cover=25.0,
            datetime_range="2022-04-01/2022-10-31",
        )


def test_sentinel2_planetary_computer_fetcher_rejects_invalid_cloud_cover() -> None:
    with pytest.raises(ValueError, match="cloud_cover must be between 0 and 100"):
        sentinel2_planetary_computer_fetcher(
            patch_size=224,
            cloud_cover=-1.0,
            datetime_range="2022-04-01/2022-10-31",
        )
    with pytest.raises(ValueError, match="cloud_cover must be between 0 and 100"):
        sentinel2_planetary_computer_fetcher(
            patch_size=224,
            cloud_cover=101.0,
            datetime_range="2022-04-01/2022-10-31",
        )


def test_sentinel2_planetary_computer_fetcher_rejects_empty_datetime_range() -> None:
    with pytest.raises(ValueError, match="datetime_range must not be empty"):
        sentinel2_planetary_computer_fetcher(
            patch_size=224,
            cloud_cover=25.0,
            datetime_range="",
        )
