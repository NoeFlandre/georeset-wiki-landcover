"""Validate project artifact directories for reproducibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd

SMALL_REQUIRED_FILES = (
    "manifest.json",
    "data/wiki/wiki_articles.json",
    "data/wiki/article_contents.json",
    "data/wiki/article_summaries.json",
    "data/wiki/article_summaries_no_place.json",
    "data/corine/synthetic_corine.geojson",
    "data/osm/osm_project_polygons.geojson",
    "data/classification/runs/small/corine_level2_summary_predictions.json",
    "data/classification/runs/small/corine_level2_summary_metrics.json",
    "data/classification/runs/small/osm_summary_predictions.json",
    "data/classification/runs/small/osm_summary_metrics.json",
)

FULL_REQUIRED_FILES = (
    "corine/bounds.json",
    "wiki/wiki_articles.json",
    "wiki/article_contents.json",
    "osm/osm_project_polygons.geojson",
)

CLASSIFICATION_RUNS = (
    ("corine_level2", "summary"),
    ("osm", "summary"),
)

PREDICTION_REQUIRED_FIELDS = {
    "pageid",
    "title",
    "target",
    "prediction",
    "prediction_labels",
    "parse_status",
    "raw_response",
    "error",
    "metadata",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, violations: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        violations.append(f"invalid JSON in {path}: {exc}")
        return None


def _required_files(root: Path, relative_paths: tuple[str, ...]) -> list[str]:
    violations: list[str] = []
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.exists():
            violations.append(f"required file missing: {relative_path}")
        elif path.is_file() and path.stat().st_size == 0:
            violations.append(f"required file is empty: {relative_path}")
    return violations


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bounds(root: Path, violations: list[str]) -> None:
    bounds = _load_json(root / "corine/bounds.json", violations)
    required_keys = ("min_lon", "min_lat", "max_lon", "max_lat")
    if not isinstance(bounds, dict) or not all(
        _is_number(bounds.get(key)) for key in required_keys
    ):
        violations.append(
            "corine/bounds.json must contain numeric min_lon, min_lat, max_lon, max_lat"
        )
        return
    if bounds["min_lon"] > bounds["max_lon"] or bounds["min_lat"] > bounds["max_lat"]:
        violations.append("corine/bounds.json min values must not exceed max values")


def _validate_wiki_article_rows(wiki_articles: Any, violations: list[str]) -> list[str]:
    if not isinstance(wiki_articles, list):
        violations.append("wiki/wiki_articles.json must be a JSON list")
        return []

    pageids: list[str] = []
    for index, article in enumerate(wiki_articles):
        if not isinstance(article, dict):
            violations.append(f"wiki article at index {index} must be a JSON object")
            continue
        pageid = article.get("pageid")
        if pageid in (None, ""):
            violations.append(f"wiki article at index {index} missing pageid")
            continue
        normalized_pageid = str(pageid)
        pageids.append(normalized_pageid)
        if not _is_number(article.get("lat")) or not _is_number(article.get("lon")):
            violations.append(f"wiki article {normalized_pageid} has non-numeric lat/lon")
    return pageids


def _validate_wiki_inputs(root: Path, violations: list[str]) -> set[str]:
    wiki_path = root / "data/wiki/wiki_articles.json"
    contents_path = root / "data/wiki/article_contents.json"
    summaries_path = root / "data/wiki/article_summaries.json"
    no_place_path = root / "data/wiki/article_summaries_no_place.json"

    wiki_articles = _load_json(wiki_path, violations)
    if not isinstance(wiki_articles, list) or not wiki_articles:
        violations.append("data/wiki/wiki_articles.json must be a non-empty JSON list")
        return set()

    pageids = _validate_wiki_article_rows(wiki_articles, violations)
    duplicates = sorted(pageid for pageid, count in Counter(pageids).items() if count > 1)
    if duplicates:
        violations.append(f"duplicate wiki pageids: {', '.join(duplicates)}")
    pageid_set = set(pageids)

    contents = _load_json(contents_path, violations)
    if not isinstance(contents, dict) or not contents:
        violations.append("data/wiki/article_contents.json must be a non-empty JSON object")
        return pageid_set

    content_keys = set(contents)
    if content_keys != pageid_set:
        violations.append(
            "article content keys must match wiki pageids: "
            f"missing={sorted(pageid_set - content_keys)} extra={sorted(content_keys - pageid_set)}"
        )

    for label, path in (
        ("article_summaries", summaries_path),
        ("article_summaries_no_place", no_place_path),
    ):
        summaries = _load_json(path, violations)
        if not isinstance(summaries, dict) or not summaries:
            violations.append(f"data/wiki/{path.name} must be a non-empty JSON object")
            continue
        summary_keys = set(summaries)
        if summary_keys != content_keys:
            violations.append(
                f"{label} keys must match article content keys: "
                f"missing={sorted(content_keys - summary_keys)} "
                f"extra={sorted(summary_keys - content_keys)}"
            )
        missing_summary = sorted(
            key
            for key, value in summaries.items()
            if not isinstance(value, dict) or not isinstance(value.get("summary"), str)
        )
        if missing_summary:
            violations.append(f"{label} records missing summary text: {missing_summary}")
    return pageid_set


def _validate_vector_file(
    root: Path,
    relative_path: str,
    *,
    required_columns: set[str],
    violations: list[str],
) -> None:
    path = root / relative_path
    try:
        frame = gpd.read_file(path)
    except (OSError, ValueError) as exc:
        violations.append(f"invalid vector artifact {relative_path}: {exc}")
        return
    if frame.empty:
        violations.append(f"vector artifact is empty: {relative_path}")
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        violations.append(f"{relative_path} missing required columns: {sorted(missing_columns)}")


def _validate_prediction_run(
    root: Path, task: str, text_source: str, violations: list[str]
) -> None:
    stem = f"{task}_{text_source}"
    run_dir = root / "data/classification/runs/small"
    predictions_path = run_dir / f"{stem}_predictions.json"
    metrics_path = run_dir / f"{stem}_metrics.json"
    predictions = _load_json(predictions_path, violations)
    metrics = _load_json(metrics_path, violations)
    if not isinstance(predictions, dict) or not predictions:
        violations.append(f"{predictions_path.relative_to(root)} must be a non-empty JSON object")
        return
    if not isinstance(metrics, dict):
        violations.append(f"{metrics_path.relative_to(root)} must be a JSON object")
        return

    ok_count = 0
    for pageid, record in predictions.items():
        if not isinstance(record, dict):
            violations.append(f"{stem} prediction {pageid} must be a JSON object")
            continue
        missing_fields = PREDICTION_REQUIRED_FIELDS - set(record)
        if missing_fields:
            violations.append(
                f"{stem} prediction {pageid} missing fields: {sorted(missing_fields)}"
            )
        if str(record.get("pageid")) != str(pageid):
            violations.append(f"{stem} prediction key/pageid mismatch for {pageid}")
        if record.get("parse_status") == "ok":
            ok_count += 1
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            violations.append(f"{stem} prediction {pageid} metadata must be a JSON object")
            continue
        for key in ("fingerprint", "text_sha256", "model", "seed", "temperature"):
            if key not in metadata:
                violations.append(f"{stem} prediction {pageid} metadata missing {key}")

    n_eligible = metrics.get("n_eligible")
    n_predicted_ok = metrics.get("n_predicted_ok")
    n_parse_error = metrics.get("n_parse_error")
    if n_eligible != len(predictions):
        violations.append(f"{stem} n_eligible={n_eligible} but predictions={len(predictions)}")
    if n_predicted_ok != ok_count:
        violations.append(f"{stem} n_predicted_ok={n_predicted_ok} but ok records={ok_count}")
    expected_parse_error = len(predictions) - ok_count
    if n_parse_error != expected_parse_error:
        violations.append(
            f"{stem} n_parse_error={n_parse_error} but non-ok records={expected_parse_error}"
        )


def _validate_manifest_hashes(root: Path, violations: list[str]) -> None:
    manifest = _load_json(root / "manifest.json", violations)
    if not isinstance(manifest, dict):
        violations.append("manifest.json must be a JSON object")
        return
    if manifest.get("mode") != "small":
        violations.append('manifest.json field "mode" must be "small"')
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        violations.append("manifest.json must include non-empty artifact_sha256 mapping")
        return
    for relative_path, expected_hash in sorted(artifact_hashes.items()):
        path = root / relative_path
        if not path.exists():
            violations.append(f"manifested artifact missing: {relative_path}")
            continue
        current_hash = sha256_file(path)
        if current_hash != expected_hash:
            violations.append(
                f"stale artifact hash for {relative_path}: "
                f"manifest={expected_hash} current={current_hash}"
            )


def _validate_small_artifacts(root: Path) -> list[str]:
    violations = _required_files(root, SMALL_REQUIRED_FILES)
    if violations:
        return violations
    _validate_manifest_hashes(root, violations)
    _validate_wiki_inputs(root, violations)
    _validate_vector_file(
        root,
        "data/corine/synthetic_corine.geojson",
        required_columns={"code_18"},
        violations=violations,
    )
    _validate_vector_file(
        root,
        "data/osm/osm_project_polygons.geojson",
        required_columns={"osm_id", "landuse", "natural"},
        violations=violations,
    )
    for task, text_source in CLASSIFICATION_RUNS:
        _validate_prediction_run(root, task, text_source, violations)
    return violations


def _validate_full_artifacts(root: Path) -> list[str]:
    violations = _required_files(root, FULL_REQUIRED_FILES)
    if violations:
        return violations

    _validate_bounds(root, violations)
    wiki_articles = _load_json(root / "wiki/wiki_articles.json", violations)
    pageids = _validate_wiki_article_rows(wiki_articles, violations)
    if pageids:
        duplicates = sorted(pageid for pageid, count in Counter(pageids).items() if count > 1)
        if duplicates:
            violations.append(f"duplicate wiki pageids: {', '.join(duplicates)}")

    contents = _load_json(root / "wiki/article_contents.json", violations)
    if isinstance(wiki_articles, list) and isinstance(contents, dict):
        wiki_pageids = set(pageids)
        stray_content_keys = sorted(set(contents) - wiki_pageids)
        if stray_content_keys:
            violations.append(
                f"article_contents has keys absent from wiki_articles: {stray_content_keys}"
            )

    _validate_vector_file(
        root,
        "osm/osm_project_polygons.geojson",
        required_columns={"osm_id"},
        violations=violations,
    )
    return violations


def validate_artifacts(root: Path | str, *, profile: str = "small") -> list[str]:
    """Return validation violations for a reproducibility artifact root."""
    root_path = Path(root)
    if not root_path.exists():
        return [f"artifact root missing: {root_path}"]
    if profile == "small":
        return _validate_small_artifacts(root_path)
    if profile == "full":
        return _validate_full_artifacts(root_path)
    return [f"unknown artifact validation profile: {profile}"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("build/reproducibility/small"),
        help="Artifact root to validate.",
    )
    parser.add_argument("--profile", choices=["small", "full"], default="small")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    violations = validate_artifacts(args.root, profile=args.profile)
    if violations:
        for violation in violations:
            print(f"ERROR: {violation}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Artifact validation passed: {args.root} ({args.profile})")


if __name__ == "__main__":
    main()
