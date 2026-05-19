"""Fetch and cache Sentinel-2 RGB patches at multiple physical window sizes."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from georeset.utils.json_io import write_csv_atomic, write_json_atomic, write_npz_atomic

NATIVE_PIXEL_SIZE_M = 10
RESIZE_METHOD = "bilinear"


class PatchFetchResult(TypedDict):
    patch: NDArray[np.uint8]
    stac_item_id: str
    eo_cloud_cover: float


PatchFetcher = Callable[[pd.Series, int, int], PatchFetchResult | None]
CoordinateTransformer = Callable[
    [str, object, list[float], list[float]], tuple[list[float], list[float]]
]


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


def patch_stats_path(output_dir: Path) -> Path:
    return output_dir / "patch_stats.csv"


def patch_validation_manifest_path(output_dir: Path) -> Path:
    return output_dir / "patch_validation_manifest.json"


def patch_contact_sheet_path(output_dir: Path) -> Path:
    return output_dir / "patch_contact_sheet.png"


def dataset_xy_from_lonlat(
    dataset: object,
    *,
    lon: float,
    lat: float,
    transform_fn: CoordinateTransformer | None = None,
) -> tuple[float, float]:
    if transform_fn is None:
        from rasterio.warp import transform

        transform_fn = transform
    crs = getattr(dataset, "crs", None)
    x_values, y_values = transform_fn("EPSG:4326", crs, [lon], [lat])
    return float(x_values[0]), float(y_values[0])


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
    patches = data["patches"]
    if len(patches) and int(patches.max()) == 0:
        raise ValueError(f"Existing Sentinel patch cache is all-zero: {path}")
    return {
        "pageids": data["pageids"].astype(str).tolist(),
        "patches": list(patches),
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


def _patch_stats(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    patches = data["patches"].astype(np.uint8)
    flat = patches.reshape(len(patches), -1) if len(patches) else np.empty((0, 0), dtype=np.uint8)
    patch_sums = flat.sum(axis=1) if len(patches) else np.array([], dtype=np.uint64)
    return {
        "file": path.name,
        "window_m": int(data["window_m"]),
        "source_pixels": int(data["source_pixels"]),
        "native_pixel_size_m": int(data["native_pixel_size_m"]),
        "resize_method": str(data["resize_method"]),
        "output_size": int(data["output_size"]),
        "n_patches": int(len(patches)),
        "all_zero_count": int((patch_sums == 0).sum()),
        "mean_pixel_value": float(patches.mean()) if len(patches) else 0.0,
        "std_pixel_value": float(patches.std()) if len(patches) else 0.0,
        "min_pixel_value": int(patches.min()) if len(patches) else 0,
        "max_pixel_value": int(patches.max()) if len(patches) else 0,
        "unique_stac_items": int(len(set(data["stac_item_id"].astype(str).tolist()))),
        "deterministic_even_window_centering": (
            f"Window(col - {int(data['source_pixels']) / 2:g}, "
            f"row - {int(data['source_pixels']) / 2:g}, "
            f"{int(data['source_pixels'])}, {int(data['source_pixels'])})"
        ),
    }


def _window_difference_stats(caches: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (left_window, left_path) in enumerate(caches):
        left = np.load(left_path, allow_pickle=False)
        left_pageids = left["pageids"].astype(str)
        left_patches = left["patches"].astype(np.uint8)
        left_map = dict(zip(left_pageids.tolist(), left_patches, strict=True))
        for right_window, right_path in caches[index + 1 :]:
            right = np.load(right_path, allow_pickle=False)
            right_pageids = right["pageids"].astype(str)
            right_patches = right["patches"].astype(np.uint8)
            right_map = dict(zip(right_pageids.tolist(), right_patches, strict=True))
            common = sorted(set(left_map) & set(right_map))
            exact_equal = sum(
                int(np.array_equal(left_map[pageid], right_map[pageid])) for pageid in common
            )
            mean_abs_diff = (
                float(
                    np.mean(
                        [
                            np.abs(
                                left_map[pageid].astype(np.int16)
                                - right_map[pageid].astype(np.int16)
                            ).mean()
                            for pageid in common
                        ]
                    )
                )
                if common
                else 0.0
            )
            rows.append(
                {
                    "file": f"{left_path.name}__vs__{right_path.name}",
                    "window_m": f"{left_window}_vs_{right_window}",
                    "source_pixels": "",
                    "native_pixel_size_m": NATIVE_PIXEL_SIZE_M,
                    "resize_method": RESIZE_METHOD,
                    "output_size": int(left["output_size"]),
                    "n_patches": int(len(common)),
                    "all_zero_count": "",
                    "mean_pixel_value": "",
                    "std_pixel_value": "",
                    "min_pixel_value": "",
                    "max_pixel_value": "",
                    "unique_stac_items": "",
                    "deterministic_even_window_centering": "",
                    "exact_equal_patch_count": int(exact_equal),
                    "mean_abs_patch_difference": mean_abs_diff,
                }
            )
    return rows


def _select_contact_sheet_pageids(
    splits: pd.DataFrame, available: set[str], n_pageids: int
) -> list[str]:
    if "label" not in splits.columns:
        return sorted(available)[:n_pageids]
    candidates = splits[splits["pageid"].astype(str).isin(available)].copy()
    candidates["pageid"] = candidates["pageid"].astype(str)
    selected: list[str] = []
    for _, group in candidates.sort_values(["label", "pageid"]).groupby("label", sort=True):
        pageid = str(group.iloc[0]["pageid"])
        if pageid not in selected:
            selected.append(pageid)
        if len(selected) >= n_pageids:
            return selected
    for pageid in sorted(available):
        if pageid not in selected:
            selected.append(pageid)
        if len(selected) >= n_pageids:
            break
    return selected


def _write_contact_sheet_atomic(
    path: Path,
    *,
    splits: pd.DataFrame,
    caches: list[tuple[int, Path]],
    n_pageids: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Contact-sheet validation requires Pillow.") from exc

    loaded = []
    available_sets = []
    for window_m, cache_path in caches:
        data = np.load(cache_path, allow_pickle=False)
        pageids = data["pageids"].astype(str)
        patches = data["patches"].astype(np.uint8)
        patch_map = dict(zip(pageids.tolist(), patches, strict=True))
        loaded.append((window_m, patch_map))
        available_sets.append(set(patch_map))
    common = set.intersection(*available_sets) if available_sets else set()
    selected = _select_contact_sheet_pageids(splits, common, n_pageids)
    if not selected:
        raise ValueError("Cannot build contact sheet without common patch pageids.")

    label_by_pageid = (
        splits.drop_duplicates("pageid").set_index("pageid")["label"].astype(str).to_dict()
        if "label" in splits.columns
        else {}
    )
    tile = 160
    header = 28
    label_width = 150
    columns = len(loaded)
    sheet = Image.new("RGB", (label_width + columns * tile, header + len(selected) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (window_m, _) in enumerate(loaded):
        draw.text((label_width + column * tile + 8, 8), f"{window_m}m", fill=(0, 0, 0))
    for row_index, pageid in enumerate(selected):
        y = header + row_index * tile
        label = label_by_pageid.get(pageid, "?")
        draw.text((8, y + 8), f"{pageid}\nlabel {label}", fill=(0, 0, 0))
        for column, (_, patch_map) in enumerate(loaded):
            patch = Image.fromarray(patch_map[pageid], mode="RGB").resize((tile, tile))
            sheet.paste(patch, (label_width + column * tile, y))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.png",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        buffer = BytesIO()
        save_image = Image.Image.__dict__["save"]
        save_image(sheet, buffer, format="PNG")
        temp_path.write_bytes(buffer.getvalue())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def write_patch_validation_artifacts(
    *,
    splits_path: Path,
    output_dir: Path,
    window_m_values: Sequence[int],
    contact_sheet_pageids: int = 12,
) -> None:
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    caches = [
        (int(window_m), output_dir / patch_filename(int(window_m))) for window_m in window_m_values
    ]
    missing = [path for _, path in caches if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing patch caches for validation: {missing}")
    rows = [_patch_stats(path) for _, path in caches]
    rows.extend(_window_difference_stats(caches))
    frame = pd.DataFrame(rows)
    all_zero_counts = pd.to_numeric(frame["all_zero_count"], errors="coerce").fillna(0).astype(int)
    if all_zero_counts.sum() > 0:
        raise ValueError("Patch validation failed: at least one all-zero patch exists.")
    std_values = pd.to_numeric(frame["std_pixel_value"], errors="coerce").dropna()
    if (std_values <= 0.0).any():
        raise ValueError(
            "Patch validation failed: at least one patch cache has zero pixel variance."
        )
    if "mean_abs_patch_difference" in frame.columns:
        diffs = pd.to_numeric(frame["mean_abs_patch_difference"], errors="coerce").dropna()
        if len(diffs) and (diffs <= 0.0).any():
            raise ValueError("Patch validation failed: at least two window caches are identical.")
    write_csv_atomic(patch_stats_path(output_dir), frame, index=False)
    write_json_atomic(
        patch_validation_manifest_path(output_dir),
        {
            "validation_artifacts": ["patch_stats.csv", "patch_contact_sheet.png"],
            "window_m_values": [int(window_m) for window_m in window_m_values],
            "native_pixel_size_m": NATIVE_PIXEL_SIZE_M,
            "source_pixels_rule": "ceil(window_m / native_pixel_size_m)",
            "resize_method": RESIZE_METHOD,
            "even_window_centering": (
                "For even source_pixels, windows are centered deterministically as "
                "Window(col - source_pixels / 2, row - source_pixels / 2, "
                "source_pixels, source_pixels)."
            ),
            "numeric_checks": {
                "all_zero_count_equals_zero": True,
                "pixel_std_positive": True,
                "window_arrays_differ": True,
            },
        },
        indent=2,
    )
    _write_contact_sheet_atomic(
        patch_contact_sheet_path(output_dir),
        splits=splits,
        caches=caches,
        n_pageids=contact_sheet_pageids,
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
            if int(result["patch"].max()) == 0:
                raise ValueError(
                    f"Fetched all-zero Sentinel patch for pageid={pageid}, window_m={window_m}"
                )
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
                x, y = dataset_xy_from_lonlat(dataset, lon=lon, lat=lat)
                row_index, col = dataset.index(x, y)
                half = source_pixels / 2.0
                window = Window(col - half, row_index - half, source_pixels, source_pixels)
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
