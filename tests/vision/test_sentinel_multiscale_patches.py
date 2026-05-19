from pathlib import Path

import numpy as np
import pandas as pd

from georeset.vision.sentinel_multiscale_patches import (
    source_pixels_for_window,
    write_multiscale_patch_caches,
)


def test_source_pixels_for_window_uses_sentinel_native_ten_meter_pixels() -> None:
    assert source_pixels_for_window(320) == 32
    assert source_pixels_for_window(325) == 33


def test_write_multiscale_patch_caches_stores_scale_metadata(tmp_path: Path) -> None:
    splits_path = tmp_path / "splits.csv"
    output_dir = tmp_path / "patches"
    pd.DataFrame([{"pageid": "1", "lat": 48.0, "lon": 2.0}]).to_csv(splits_path, index=False)

    def fetcher(row: pd.Series, window_m: int, output_size: int):
        assert row["pageid"] == "1"
        return {
            "patch": np.ones((output_size, output_size, 3), dtype=np.uint8) * (window_m // 320),
            "stac_item_id": f"item-{window_m}",
            "eo_cloud_cover": 12.5,
        }

    write_multiscale_patch_caches(
        splits_path=splits_path,
        output_dir=output_dir,
        window_m_values=[320, 2240],
        output_size=8,
        fetcher=fetcher,
    )

    cache = np.load(output_dir / "sentinel_rgb_window_0320m.npz")
    assert cache["pageids"].tolist() == ["1"]
    assert int(cache["window_m"]) == 320
    assert int(cache["source_pixels"]) == 32
    assert int(cache["native_pixel_size_m"]) == 10
    assert str(cache["resize_method"]) == "bilinear"
    assert cache["stac_item_id"].tolist() == ["item-320"]
    assert cache["eo_cloud_cover"].tolist() == [12.5]
