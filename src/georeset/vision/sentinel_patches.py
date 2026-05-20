"""Sentinel-2 RGB patch caching for CLIP experiments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from georeset.utils.json_io import write_npz_atomic

RgbPatch = NDArray[np.uint8]
PatchFetcher = Callable[[pd.Series], RgbPatch | None]


def write_patch_cache(
    *,
    splits_path: Path,
    output_path: Path,
    fetcher: PatchFetcher,
) -> None:
    rows = pd.read_csv(splits_path, dtype={"pageid": str})
    unique_rows = rows.drop_duplicates("pageid").sort_values("pageid")
    if output_path.exists():
        cached = np.load(output_path)
        pageids = cached["pageids"].astype(str).tolist()
        patches = [np.asarray(patch, dtype=np.uint8) for patch in cached["patches"]]
    else:
        pageids = []
        patches = []
    fetched_pageids = set(pageids)
    for index, (_, row) in enumerate(unique_rows.iterrows(), start=1):
        pageid = str(row["pageid"])
        if pageid in fetched_pageids:
            continue
        try:
            patch = fetcher(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{len(unique_rows)}] {pageid}: skipped ({exc})", flush=True)
            continue
        if patch is None:
            print(f"[{index}/{len(unique_rows)}] {pageid}: no patch", flush=True)
            continue
        if patch.dtype != np.uint8 or patch.ndim != 3 or patch.shape[-1] != 3:
            raise ValueError("patch fetcher must return uint8 RGB arrays")
        pageids.append(pageid)
        patches.append(patch)
        write_npz_atomic(output_path, pageids=np.array(pageids), patches=np.stack(patches))
        print(f"[{index}/{len(unique_rows)}] {pageid}: cached", flush=True)
    if not patches:
        raise ValueError("No Sentinel patches were fetched.")
    write_npz_atomic(output_path, pageids=np.array(pageids), patches=np.stack(patches))


def sentinel2_planetary_computer_fetcher(
    *,
    patch_size: int,
    cloud_cover: float,
    datetime_range: str,
) -> PatchFetcher:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if cloud_cover < 0.0 or cloud_cover > 100.0:
        raise ValueError("cloud_cover must be between 0 and 100")
    if not datetime_range.strip():
        raise ValueError("datetime_range must not be empty")

    try:
        import planetary_computer
        import pystac_client
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "Sentinel patch fetching requires optional vision dependencies. "
            "Run with `uv run --group vision ...`."
        ) from exc

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    half = patch_size // 2

    def _scale_rgb(bands: list[NDArray[np.integer]]) -> RgbPatch:
        stacked = np.stack(bands, axis=-1).astype(np.float32)
        scaled = np.clip(stacked / 3000.0, 0.0, 1.0) * 255.0
        return np.asarray(scaled.astype(np.uint8))

    def fetch(row: pd.Series) -> RgbPatch | None:
        lon = float(row["lon"])
        lat = float(row["lat"])
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": cloud_cover}},
            limit=5,
        )
        items = sorted(
            search.items(),
            key=lambda item: float(item.properties.get("eo:cloud_cover", 100.0)),
        )
        if not items:
            return None
        item = planetary_computer.sign(items[0])
        arrays = []
        for asset_key in ("B04", "B03", "B02"):
            href = item.assets[asset_key].href
            with (
                rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_HTTP_TIMEOUT="30",
                    GDAL_HTTP_MAX_RETRY="2",
                ),
                rasterio.open(href) as dataset,
            ):
                x, y = transform("EPSG:4326", dataset.crs, [lon], [lat])
                row_index, col_index = dataset.index(x[0], y[0])
                window = Window(col_index - half, row_index - half, patch_size, patch_size)
                array = dataset.read(
                    1,
                    window=window,
                    boundless=True,
                    fill_value=0,
                    out_shape=(patch_size, patch_size),
                    resampling=Resampling.bilinear,
                )
                arrays.append(array)
        return _scale_rgb(arrays)

    return fetch
