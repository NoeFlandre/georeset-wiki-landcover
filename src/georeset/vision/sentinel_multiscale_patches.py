"""Fetch and cache Sentinel-2 RGB patches at multiple physical window sizes."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from georeset.utils.json_io import write_npz_atomic

NATIVE_PIXEL_SIZE_M = 10
RESIZE_METHOD = "bilinear"


class PatchFetchResult(TypedDict):
    patch: NDArray[np.uint8]
    stac_item_id: str
    eo_cloud_cover: float


PatchFetcher = Callable[[pd.Series, int, int], PatchFetchResult | None]


def source_pixels_for_window(
    window_m: int, *, native_pixel_size_m: int = NATIVE_PIXEL_SIZE_M
) -> int:
    if window_m <= 0:
        raise ValueError("window_m must be positive")
    if native_pixel_size_m <= 0:
        raise ValueError("native_pixel_size_m must be positive")
    return int(math.ceil(window_m / float(native_pixel_size_m)))


def patch_filename(window_m: int) -> str:
    return f"sentinel_rgb_window_{window_m:04d}m.npz"


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "pageids": [],
            "patches": [],
            "lat": [],
            "lon": [],
            "stac_item_id": [],
            "eo_cloud_cover": [],
        }
    data = np.load(path, allow_pickle=False)
    return {
        "pageids": data["pageids"].astype(str).tolist(),
        "patches": list(data["patches"]),
        "lat": data["lat"].astype(float).tolist(),
        "lon": data["lon"].astype(float).tolist(),
        "stac_item_id": data["stac_item_id"].astype(str).tolist(),
        "eo_cloud_cover": data["eo_cloud_cover"].astype(float).tolist(),
    }


def _write_cache(path: Path, rows: dict[str, Any], *, window_m: int, output_size: int) -> None:
    write_npz_atomic(
        path,
        pageids=np.asarray(rows["pageids"], dtype=str),
        patches=np.asarray(rows["patches"], dtype=np.uint8),
        lat=np.asarray(rows["lat"], dtype=float),
        lon=np.asarray(rows["lon"], dtype=float),
        window_m=np.asarray(window_m, dtype=np.int64),
        source_pixels=np.asarray(source_pixels_for_window(window_m), dtype=np.int64),
        native_pixel_size_m=np.asarray(NATIVE_PIXEL_SIZE_M, dtype=np.int64),
        resize_method=np.asarray(RESIZE_METHOD),
        output_size=np.asarray(output_size, dtype=np.int64),
        stac_item_id=np.asarray(rows["stac_item_id"], dtype=str),
        eo_cloud_cover=np.asarray(rows["eo_cloud_cover"], dtype=float),
    )


def write_multiscale_patch_caches(
    *,
    splits_path: Path,
    output_dir: Path,
    window_m_values: Sequence[int],
    output_size: int,
    fetcher: PatchFetcher,
) -> None:
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    frame = pd.read_csv(splits_path, dtype={"pageid": str})
    unique = frame.drop_duplicates("pageid").sort_values("pageid").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for window_m in window_m_values:
        path = output_dir / patch_filename(window_m)
        cache = _load_existing(path)
        seen = set(cache["pageids"])
        for _, row in unique.iterrows():
            pageid = str(row["pageid"])
            if pageid in seen:
                continue
            result = fetcher(row, int(window_m), output_size)
            if result is None:
                continue
            cache["pageids"].append(pageid)
            cache["patches"].append(result["patch"].astype(np.uint8))
            cache["lat"].append(float(row["lat"]))
            cache["lon"].append(float(row["lon"]))
            cache["stac_item_id"].append(str(result["stac_item_id"]))
            cache["eo_cloud_cover"].append(float(result["eo_cloud_cover"]))
            seen.add(pageid)
            _write_cache(path, cache, window_m=int(window_m), output_size=output_size)
        if cache["pageids"]:
            _write_cache(path, cache, window_m=int(window_m), output_size=output_size)


def resize_rgb_patch(patch: NDArray[np.uint8], output_size: int) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("RGB patch resizing requires `uv run --group vision ...`.") from exc

    image = Image.fromarray(patch.astype(np.uint8), mode="RGB")
    resized = image.resize((output_size, output_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def sentinel2_planetary_computer_multiscale_fetcher(
    *,
    cloud_cover: float,
    datetime_range: str,
) -> PatchFetcher:
    if cloud_cover < 0.0 or cloud_cover > 100.0:
        raise ValueError("cloud_cover must be between 0 and 100")
    if not datetime_range:
        raise ValueError("datetime_range must not be empty")

    def fetch(row: pd.Series, window_m: int, output_size: int) -> PatchFetchResult | None:
        try:
            import planetary_computer
            import pystac_client
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.windows import Window
        except ImportError as exc:
            raise RuntimeError(
                "Sentinel patch fetching requires the optional vision dependencies. "
                "Run with `uv run --group vision ...`."
            ) from exc

        lat = float(row["lat"])
        lon = float(row["lon"])
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": cloud_cover}},
            limit=10,
        )
        items = sorted(
            search.items(), key=lambda item: float(item.properties.get("eo:cloud_cover", 100.0))
        )
        if not items:
            return None
        item = planetary_computer.sign(items[0])
        arrays = []
        source_pixels = source_pixels_for_window(window_m)
        for band in ("B04", "B03", "B02"):
            href = item.assets[band].href
            with rasterio.open(href) as dataset:
                row_col, col = dataset.index(lon, lat)
                half = source_pixels / 2.0
                window = Window(col - half, row_col - half, source_pixels, source_pixels)
                band_array = dataset.read(
                    1,
                    window=window,
                    boundless=True,
                    fill_value=0,
                    out_shape=(output_size, output_size),
                    resampling=Resampling.bilinear,
                )
                arrays.append(band_array)
        stacked = np.stack(arrays, axis=-1)
        rgb = (np.clip(stacked / 3000.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        return {
            "patch": rgb,
            "stac_item_id": str(item.id),
            "eo_cloud_cover": float(item.properties.get("eo:cloud_cover", np.nan)),
        }

    return fetch
