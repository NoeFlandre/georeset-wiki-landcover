"""Compute CORINE spatial-confidence diagnostics for frozen prediction pageids."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import Point

from src.spatial.corine_confidence import (
    METRIC_CRS,
    compute_article_spatial_confidence,
    derive_level2_label,
    prepare_corine_gdf,
)

EXPERIMENT_ID = "corine_spatial_confidence_v1"
PARENT_EXPERIMENT_ID = "article_text_classification_e2e_with_shuffled_control_v1"
DEFAULT_PARENT_DIR = Path("data/experiments/article_text_classification_e2e_with_shuffled_control_v1")
DEFAULT_OUTPUT_DIR = Path("data/experiments/corine_spatial_confidence_v1")
DEFAULT_RADII = [100, 250, 500, 1000]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--wiki-articles-path", type=Path, default=Path("data/wiki/wiki_articles.json"))
    parser.add_argument(
        "--corine-polygons-path",
        type=Path,
        default=Path("data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radii", type=int, nargs="+", default=DEFAULT_RADII)
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _prediction_files(parent_dir: Path) -> list[Path]:
    return sorted(parent_dir.glob("*_predictions.json"))


def load_parent_prediction_pageids(parent_dir: Path) -> set[str]:
    pageids: set[str] = set()
    for path in _prediction_files(parent_dir):
        pageids.update(str(pageid) for pageid in _load_json(path))
    return pageids


def load_corine_targets(parent_dir: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    seen_by_pageid: dict[str, set[str]] = {}
    for path in sorted(parent_dir.glob("corine_level2_*_predictions.json")):
        for pageid, record in _load_json(path).items():
            target = str(record["target"])
            seen_by_pageid.setdefault(str(pageid), set()).add(target)
    mismatches = {pageid: values for pageid, values in seen_by_pageid.items() if len(values) > 1}
    if mismatches:
        raise ValueError(f"Inconsistent CORINE targets across parent runs: {mismatches}")
    for pageid, values in seen_by_pageid.items():
        targets[pageid] = next(iter(values))
    return targets


def articles_to_points(articles: list[dict[str, Any]], pageids: set[str]) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    geometries: list[Point] = []
    for article in articles:
        pageid = str(article.get("pageid"))
        lat = article.get("lat")
        lon = article.get("lon")
        if pageid not in pageids or not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        rows.append({"pageid": pageid, "title": article.get("title", pageid)})
        geometries.append(Point(float(lon), float(lat)))
    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


def derive_point_labels(
    articles_gdf: gpd.GeoDataFrame, corine_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    articles = articles_gdf.to_crs(METRIC_CRS)
    corine = prepare_corine_gdf(corine_gdf, label_col="label")
    joined = gpd.sjoin(
        articles[["pageid", "title", "geometry"]],
        corine[["label", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    labels_by_pageid: dict[str, set[str]] = {}
    for pageid, group in joined.groupby("pageid", sort=False):
        labels_by_pageid[str(pageid)] = set(group["label"].astype(str))

    rows: list[dict[str, Any]] = []
    geometries = []
    for _, article in articles.iterrows():
        labels = labels_by_pageid.get(str(article["pageid"]), set())
        if len(labels) != 1:
            continue
        rows.append(
            {
                "pageid": str(article["pageid"]),
                "title": article.get("title", str(article["pageid"])),
                "point_label": next(iter(labels)),
            }
        )
        geometries.append(make_valid(article.geometry))
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=METRIC_CRS)


def validate_corine_targets(point_labels: pd.DataFrame, corine_targets: dict[str, str]) -> None:
    mismatches = []
    labels = dict(
        zip(point_labels["pageid"].astype(str), point_labels["point_label"].astype(str), strict=False)
    )
    for pageid, target in corine_targets.items():
        derived = labels.get(pageid)
        if derived is not None and derived != target:
            mismatches.append({"pageid": pageid, "derived_point_label": derived, "corine_target": target})
    if mismatches:
        raise ValueError(f"CORINE target mismatch against derived full-CORINE point labels: {mismatches[:10]}")


def _add_artificial_shares(confidence: pd.DataFrame, radii: list[int]) -> pd.DataFrame:
    for radius in radii:
        column = f"label_shares_{radius}m"
        if column not in confidence.columns:
            continue
        confidence[f"artificial_share_{radius}m"] = confidence[column].map(
            lambda shares: sum(value for label, value in shares.items() if str(label).startswith("1"))
        )
    return confidence


def _write_outputs(
    confidence: pd.DataFrame,
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_df = confidence.copy()
    for column in csv_df.columns:
        if column.startswith("label_shares_"):
            csv_df[column] = csv_df[column].map(lambda value: json.dumps(value, sort_keys=True))
    csv_df.to_csv(output_dir / "spatial_confidence.csv", index=False)
    try:
        confidence.to_parquet(output_dir / "spatial_confidence.parquet", index=False)
        manifest["parquet_written"] = True
    except Exception as exc:
        manifest["parquet_written"] = False
        manifest["parquet_error"] = str(exc)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# CORINE Spatial Confidence v1",
                "",
                f"- articles with confidence rows: {manifest['number_of_articles_with_valid_confidence_values']}",
                f"- radii_m: {manifest['radii_m']}",
                f"- CRS used: {manifest['crs_used']}",
                "- Full CORINE was used, including artificial classes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pageids = load_parent_prediction_pageids(args.parent_experiment_dir)
    corine_targets = load_corine_targets(args.parent_experiment_dir)
    articles = articles_to_points(_load_json(args.wiki_articles_path), pageids)
    corine = gpd.read_file(args.corine_polygons_path)
    if "label" not in corine.columns:
        corine["label"] = corine["code_18"].map(derive_level2_label)
    articles_with_labels = derive_point_labels(articles, corine)
    validate_corine_targets(articles_with_labels, corine_targets)

    confidence = compute_article_spatial_confidence(
        articles_with_labels,
        corine,
        args.radii,
        point_label_col="point_label",
        pageid_col="pageid",
    )
    confidence = _add_artificial_shares(confidence, args.radii)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "radii_m": args.radii,
        "crs_used": METRIC_CRS,
        "used_full_corine": True,
        "input_paths": {
            "parent_experiment_dir": str(args.parent_experiment_dir),
            "wiki_articles_path": str(args.wiki_articles_path),
            "corine_polygons_path": str(args.corine_polygons_path),
        },
        "number_of_articles": len(pageids),
        "number_of_articles_with_valid_confidence_values": len(confidence),
        "code_config_parameters": {
            "point_labels": "derived from full CORINE point-in-polygon",
            "corine_target_validation": "strict where parent CORINE target exists",
        },
    }
    _write_outputs(confidence, args.output_dir, manifest)


if __name__ == "__main__":
    main()
