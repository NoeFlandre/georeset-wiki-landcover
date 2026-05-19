from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from georeset.vision.sentinel_multiscale_patches import (
    dataset_xy_from_lonlat,
    patch_contact_sheet_path,
    patch_stats_path,
    patch_validation_manifest_path,
    source_pixels_for_window,
    write_multiscale_patch_caches,
    write_patch_validation_artifacts,
)


def test_source_pixels_for_window_uses_sentinel_native_ten_meter_pixels() -> None:
    assert source_pixels_for_window(320) == 32
    assert source_pixels_for_window(325) == 33


def test_patch_validation_artifact_paths_use_expected_names(tmp_path: Path) -> None:
    assert patch_stats_path(tmp_path) == tmp_path / "patch_stats.csv"
    assert patch_validation_manifest_path(tmp_path) == tmp_path / "patch_validation_manifest.json"
    assert patch_contact_sheet_path(tmp_path) == tmp_path / "patch_contact_sheet.png"


def test_dataset_xy_from_lonlat_reprojects_to_dataset_crs() -> None:
    class FakeDataset:
        crs = "EPSG:2154"

    calls = []

    def transform_fn(src_crs, dst_crs, xs, ys):
        calls.append((src_crs, dst_crs, xs, ys))
        return [654321.0], [6865432.0]

    x, y = dataset_xy_from_lonlat(FakeDataset(), lon=2.35, lat=48.85, transform_fn=transform_fn)

    assert (x, y) == (654321.0, 6865432.0)
    assert calls == [("EPSG:4326", "EPSG:2154", [2.35], [48.85])]


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


def test_write_patch_validation_artifacts_creates_stats_manifest_and_contact_sheet(
    tmp_path: Path,
) -> None:
    splits_path = tmp_path / "splits.csv"
    output_dir = tmp_path / "patches"
    pd.DataFrame(
        [
            {"pageid": "1", "lat": 48.0, "lon": 2.0, "label": "21"},
            {"pageid": "2", "lat": 49.0, "lon": 3.0, "label": "22"},
        ]
    ).to_csv(splits_path, index=False)

    def fetcher(row: pd.Series, window_m: int, output_size: int):
        page_offset = int(row["pageid"])
        return {
            "patch": np.ones((output_size, output_size, 3), dtype=np.uint8)
            * (window_m // 320 + page_offset),
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
    write_patch_validation_artifacts(
        splits_path=splits_path,
        output_dir=output_dir,
        window_m_values=[320, 2240],
        contact_sheet_pageids=2,
    )

    stats = pd.read_csv(output_dir / "patch_stats.csv")
    cache_rows = stats[~stats["file"].str.contains("__vs__")]
    assert set(cache_rows["window_m"].astype(str)) == {"320", "2240"}
    assert cache_rows["all_zero_count"].astype(int).sum() == 0
    assert set(cache_rows["source_pixels"].astype(int)) == {32, 224}
    difference_rows = stats[stats["file"].str.contains("__vs__")]
    assert difference_rows["mean_abs_patch_difference"].astype(float).min() > 0.0
    assert (output_dir / "patch_contact_sheet.png").exists()
    assert (output_dir / "patch_validation_manifest.json").exists()


def test_write_multiscale_patch_caches_rejects_all_zero_patches(tmp_path: Path) -> None:
    splits_path = tmp_path / "splits.csv"
    output_dir = tmp_path / "patches"
    pd.DataFrame([{"pageid": "1", "lat": 48.0, "lon": 2.0}]).to_csv(splits_path, index=False)

    def fetcher(row: pd.Series, window_m: int, output_size: int):
        return {
            "patch": np.zeros((output_size, output_size, 3), dtype=np.uint8),
            "stac_item_id": f"item-{window_m}",
            "eo_cloud_cover": 12.5,
        }

    with pytest.raises(ValueError, match="all-zero Sentinel patch"):
        write_multiscale_patch_caches(
            splits_path=splits_path,
            output_dir=output_dir,
            window_m_values=[320],
            output_size=8,
            fetcher=fetcher,
        )


def test_write_multiscale_patch_caches_rejects_existing_all_zero_cache(tmp_path: Path) -> None:
    splits_path = tmp_path / "splits.csv"
    output_dir = tmp_path / "patches"
    output_dir.mkdir()
    pd.DataFrame([{"pageid": "1", "lat": 48.0, "lon": 2.0}]).to_csv(splits_path, index=False)
    np.savez(
        output_dir / "sentinel_rgb_window_0320m.npz",
        pageids=np.array(["1"]),
        patches=np.zeros((1, 8, 8, 3), dtype=np.uint8),
        lat=np.array([48.0]),
        lon=np.array([2.0]),
        stac_item_id=np.array(["bad"]),
        eo_cloud_cover=np.array([0.0]),
    )

    def fetcher(row: pd.Series, window_m: int, output_size: int):
        return {
            "patch": np.ones((output_size, output_size, 3), dtype=np.uint8),
            "stac_item_id": f"item-{window_m}",
            "eo_cloud_cover": 12.5,
        }

    with pytest.raises(ValueError, match="Existing Sentinel patch cache is all-zero"):
        write_multiscale_patch_caches(
            splits_path=splits_path,
            output_dir=output_dir,
            window_m_values=[320],
            output_size=8,
            fetcher=fetcher,
        )
