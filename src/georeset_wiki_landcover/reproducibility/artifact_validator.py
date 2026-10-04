"""Validate project artifact directories for reproducibility checks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import box

from georeset_wiki_landcover.classification import runner as classification_runner

SMALL_SYNTHETIC_INPUTS = (
    "data/wiki/wiki_articles.json",
    "data/wiki/article_contents.json",
    "data/wiki/article_summaries.json",
    "data/wiki/article_summaries_no_place.json",
    "data/corine/synthetic_corine.geojson",
    "data/osm/osm_project_polygons.geojson",
)

SMALL_SYNTHETIC_OUTPUTS = (
    "data/classification/runs/small/corine_level2_summary_predictions.json",
    "data/classification/runs/small/corine_level2_summary_metrics.json",
    "data/classification/runs/small/osm_summary_predictions.json",
    "data/classification/runs/small/osm_summary_metrics.json",
)

SMALL_REQUIRED_FILES = (
    "manifest.json",
    *SMALL_SYNTHETIC_INPUTS,
    *SMALL_SYNTHETIC_OUTPUTS,
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
SMALL_SYNTHETIC_PAGEIDS = frozenset({"100", "200"})
SMALL_SYNTHETIC_MODEL = "synthetic-deterministic-classifier"
SMALL_SYNTHETIC_SEED = 42
SMALL_SYNTHETIC_TEMPERATURE = 0.0
SMALL_SYNTHETIC_TITLES = {
    "100": "Synthetic Forest",
    "200": "Synthetic Meadow",
}
SMALL_SYNTHETIC_SUMMARIES = {
    "100": "Synthetic forest land cover with trees.",
    "200": "Synthetic meadow land cover with grass.",
}
SMALL_SYNTHETIC_ARTICLE_CONTENTS = {
    "100": {
        "title": "Synthetic Forest",
        "content": "A synthetic forest article for reproducibility smoke tests.",
        "url": "https://example.invalid/wiki/Synthetic_Forest",
    },
    "200": {
        "title": "Synthetic Meadow",
        "content": "A synthetic meadow article for reproducibility smoke tests.",
        "url": "https://example.invalid/wiki/Synthetic_Meadow",
    },
}
SMALL_SYNTHETIC_SUMMARIES_NO_PLACE = {
    "100": "A forest-like area with trees.",
    "200": "A meadow-like area with grass.",
}
SMALL_SYNTHETIC_PROMPT = "Synthetic deterministic classifier; no prompt sent to an LLM."
SMALL_SYNTHETIC_SYSTEM_PROMPT = "Synthetic deterministic classifier."
SMALL_SYNTHETIC_COORDINATES = {
    "100": (0.5, 0.5),
    "200": (0.5, 2.5),
}
SMALL_SYNTHETIC_WIKI_ARTICLES = {
    "100": {
        "pageid": 100,
        "title": SMALL_SYNTHETIC_TITLES["100"],
        "lat": SMALL_SYNTHETIC_COORDINATES["100"][0],
        "lon": SMALL_SYNTHETIC_COORDINATES["100"][1],
        "url": "https://example.invalid/wiki/Synthetic_Forest",
    },
    "200": {
        "pageid": 200,
        "title": SMALL_SYNTHETIC_TITLES["200"],
        "lat": SMALL_SYNTHETIC_COORDINATES["200"][0],
        "lon": SMALL_SYNTHETIC_COORDINATES["200"][1],
        "url": "https://example.invalid/wiki/Synthetic_Meadow",
    },
}
SMALL_SYNTHETIC_VECTOR_FEATURES: dict[str, dict[str, Any]] = {
    "data/corine/synthetic_corine.geojson": {
        "id_field": "ID",
        "features": {
            "1": {"code_18": "311", "bounds": (0, 0, 1, 1)},
            "2": {"code_18": "211", "bounds": (2, 0, 3, 1)},
        },
    },
    "data/osm/osm_project_polygons.geojson": {
        "id_field": "osm_id",
        "features": {
            "synthetic/wood": {
                "landuse": None,
                "natural": "wood",
                "bounds": (0, 0, 1, 1),
            },
            "synthetic/meadow": {
                "landuse": "meadow",
                "natural": None,
                "bounds": (2, 0, 3, 1),
            },
        },
    },
}
SMALL_SYNTHETIC_COUNTS = {
    "wiki_articles": 2,
    "corine_level2_summary_predictions": 2,
    "osm_summary_predictions": 2,
}
SMALL_SYNTHETIC_EXPECTED_PREDICTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "corine_level2_summary": {
        "100": {
            "pageid": "100",
            "title": SMALL_SYNTHETIC_TITLES["100"],
            "target": "31",
            "prediction": "31",
            "prediction_labels": ["31"],
            "parse_status": "ok",
            "raw_response": '{"label": "31"}',
            "error": None,
        },
        "200": {
            "pageid": "200",
            "title": SMALL_SYNTHETIC_TITLES["200"],
            "target": "21",
            "prediction": "21",
            "prediction_labels": ["21"],
            "parse_status": "ok",
            "raw_response": '{"label": "21"}',
            "error": None,
        },
    },
    "osm_summary": {
        "100": {
            "pageid": "100",
            "title": SMALL_SYNTHETIC_TITLES["100"],
            "target": ["wood"],
            "prediction": ["wood"],
            "prediction_labels": ["wood"],
            "parse_status": "ok",
            "raw_response": "{\"labels\": ['wood']}",
            "error": None,
        },
        "200": {
            "pageid": "200",
            "title": SMALL_SYNTHETIC_TITLES["200"],
            "target": ["meadow"],
            "prediction": ["meadow"],
            "prediction_labels": ["meadow"],
            "parse_status": "ok",
            "raw_response": "{\"labels\": ['meadow']}",
            "error": None,
        },
    },
}
SMALL_SYNTHETIC_KNOWN_METRICS: dict[str, dict[str, Any]] = {
    "corine_level2_summary_metrics": {
        "n_eligible": 2,
        "n_predicted_ok": 2,
        "n_parse_error": 0,
        "coverage": 1.0,
        "accuracy": 1.0,
        "accuracy_including_parse_errors_as_wrong": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_recall_including_parse_errors_as_wrong": 1.0,
        "macro_f1": 1.0,
        "macro_f1_including_parse_errors_as_wrong": 1.0,
        "per_label": {
            "21": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
            "31": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        },
        "task": "corine_level2",
        "text_source": "summary",
        "allowed_labels": ["21", "31"],
        "labels_evaluated": ["21", "31"],
    },
    "osm_summary_metrics": {
        "n_eligible": 2,
        "n_predicted_ok": 2,
        "n_parse_error": 0,
        "coverage": 1.0,
        "exact_match_accuracy": 1.0,
        "exact_match_accuracy_including_parse_errors_as_empty": 1.0,
        "micro_precision": 1.0,
        "micro_recall": 1.0,
        "micro_f1": 1.0,
        "micro_f1_including_parse_errors_as_empty": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "macro_f1": 1.0,
        "macro_f1_including_parse_errors_as_empty": 1.0,
        "per_label": {
            "meadow": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
            "wood": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        },
        "task": "osm",
        "text_source": "summary",
        "allowed_labels": [
            "allotments",
            "bare_rock",
            "beach",
            "farmland",
            "farmyard",
            "forest",
            "grass",
            "grassland",
            "greenhouse_horticulture",
            "heath",
            "meadow",
            "mud",
            "orchard",
            "plant_nursery",
            "sand",
            "scree",
            "scrub",
            "shingle",
            "vineyard",
            "water",
            "wetland",
            "wood",
        ],
        "labels_evaluated": ["meadow", "wood"],
    },
}

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


def _matches_expected_json(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_matches_expected_json(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _matches_expected_json(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    if _is_number(expected):
        return _is_number(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


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
    required_hashes = set(SMALL_REQUIRED_FILES) - {"manifest.json"}
    missing_hashes = sorted(required_hashes - set(artifact_hashes))
    if missing_hashes:
        violations.append(
            f"manifest artifact_sha256 missing required artifact hashes: {missing_hashes}"
        )
    unexpected_hashes = sorted(set(artifact_hashes) - required_hashes)
    if unexpected_hashes:
        violations.append(
            f"manifest artifact_sha256 contains undeclared artifact hashes: {unexpected_hashes}"
        )
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


def _validate_small_synthetic_inputs(root: Path, violations: list[str]) -> None:
    wiki_articles = _load_json(root / "data/wiki/wiki_articles.json", violations)
    if isinstance(wiki_articles, list):
        relative_path = "data/wiki/wiki_articles.json"
        for article in wiki_articles:
            if not isinstance(article, dict):
                continue
            pageid = str(article.get("pageid"))
            expected_article = SMALL_SYNTHETIC_WIKI_ARTICLES.get(pageid)
            if expected_article is None:
                continue
            differing_fields = sorted(
                field
                for field in article.keys() | expected_article.keys()
                if (
                    field not in article
                    or field not in expected_article
                    or not _matches_expected_json(article[field], expected_article[field])
                )
            )
            if "title" in differing_fields:
                violations.append(
                    f"synthetic wiki article {pageid} title={article.get('title')!r}; "
                    f"expected {expected_article['title']!r}"
                )
            if "lat" in differing_fields or "lon" in differing_fields:
                actual_coordinates = (article.get("lat"), article.get("lon"))
                expected_coordinates = (expected_article["lat"], expected_article["lon"])
                violations.append(
                    f"synthetic wiki article {pageid} coordinates={actual_coordinates!r}; "
                    f"expected {expected_coordinates!r}"
                )
            other_differences = sorted(set(differing_fields) - {"title", "lat", "lon"})
            if other_differences:
                violations.append(
                    f"synthetic {relative_path} record {pageid} differs from the deterministic "
                    f"fixture for fields {other_differences}"
                )

    for relative_path, contract in SMALL_SYNTHETIC_VECTOR_FEATURES.items():
        path = root / relative_path
        try:
            frame = gpd.read_file(path)
        except (OSError, ValueError) as exc:
            violations.append(f"invalid synthetic vector artifact {relative_path}: {exc}")
            continue

        expected_features = contract["features"]
        id_field = contract["id_field"]
        if len(frame) != len(expected_features):
            violations.append(
                f"synthetic {relative_path} feature count={len(frame)}; "
                f"expected {len(expected_features)}"
            )
        if id_field not in frame.columns:
            violations.append(f"synthetic {relative_path} missing identity field {id_field}")
            continue

        expected_attribute_fields = {id_field}
        for expected_feature in expected_features.values():
            expected_attribute_fields.update(
                field for field in expected_feature if field != "bounds"
            )
        actual_attribute_fields = set(frame.columns) - {frame.geometry.name}
        if actual_attribute_fields != expected_attribute_fields:
            missing_fields = sorted(expected_attribute_fields - actual_attribute_fields)
            unexpected_fields = sorted(actual_attribute_fields - expected_attribute_fields)
            violations.append(
                f"synthetic {relative_path} attribute fields differ from the deterministic "
                f"fixture: missing={missing_fields} unexpected={unexpected_fields}"
            )

        if frame.crs is None or frame.crs.to_epsg() != 4326:
            violations.append(f"synthetic {relative_path} CRS must be EPSG:4326")

        for identity, expected in expected_features.items():
            selected = frame.loc[frame[id_field].astype(str) == identity]
            if len(selected) != 1:
                violations.append(
                    f"synthetic {relative_path} must contain exactly one feature {identity}"
                )
                continue
            feature = selected.iloc[0]
            for field, expected_value in expected.items():
                if field == "bounds":
                    expected_geometry = box(*expected_value)
                    if feature.geometry is None or not feature.geometry.equals(expected_geometry):
                        violations.append(
                            f"synthetic {relative_path} feature {identity} geometry differs "
                            "from the deterministic fixture"
                        )
                elif field not in frame.columns:
                    continue
                elif expected_value is None:
                    if not bool(selected[field].isna().iloc[0]):
                        violations.append(
                            f"synthetic {relative_path} feature {identity} {field}="
                            f"{feature[field]!r}; expected null"
                        )
                elif feature[field] != expected_value:
                    violations.append(
                        f"synthetic {relative_path} feature {identity} {field}="
                        f"{feature[field]!r}; expected {expected_value!r}"
                    )


def _expected_small_synthetic_metadata(task: str, pageid: str) -> dict[str, Any]:
    allowed_labels = SMALL_SYNTHETIC_KNOWN_METRICS[f"{task}_summary_metrics"]["allowed_labels"]
    return {
        "task": task,
        "text_source": "summary",
        "model": SMALL_SYNTHETIC_MODEL,
        "model_repo_id": None,
        "seed": SMALL_SYNTHETIC_SEED,
        "temperature": SMALL_SYNTHETIC_TEMPERATURE,
        "allowed_labels": allowed_labels,
        "prompt": SMALL_SYNTHETIC_PROMPT,
        "system_prompt": SMALL_SYNTHETIC_SYSTEM_PROMPT,
        "attempt_count": 1,
        "fingerprint": classification_runner.prediction_fingerprint(
            task,
            "summary",
            SMALL_SYNTHETIC_MODEL,
            None,
            SMALL_SYNTHETIC_SEED,
            SMALL_SYNTHETIC_TEMPERATURE,
            allowed_labels,
        ),
        "text_sha256": classification_runner.text_fingerprint(SMALL_SYNTHETIC_SUMMARIES[pageid]),
    }


def _validate_small_synthetic_summaries(root: Path, violations: list[str]) -> None:
    summaries = _load_json(root / "data/wiki/article_summaries.json", violations)
    if not isinstance(summaries, dict):
        return
    if set(summaries) != SMALL_SYNTHETIC_PAGEIDS:
        missing = sorted(SMALL_SYNTHETIC_PAGEIDS - set(summaries))
        extra = sorted(set(summaries) - SMALL_SYNTHETIC_PAGEIDS)
        violations.append(
            f"synthetic article summaries must have exactly the expected pageids: "
            f"missing={missing} extra={extra}"
        )
    for pageid, expected_summary in SMALL_SYNTHETIC_SUMMARIES.items():
        record = summaries.get(pageid)
        if not _matches_expected_json(record, {"summary": expected_summary}):
            actual = record.get("summary") if isinstance(record, dict) else record
            violations.append(
                f"synthetic article summary {pageid}={actual!r}; expected {expected_summary!r}"
            )


def _validate_small_synthetic_contents(root: Path, violations: list[str]) -> None:
    relative_path = "data/wiki/article_contents.json"
    contents = _load_json(root / relative_path, violations)
    if not isinstance(contents, dict):
        return
    for pageid, expected_content in SMALL_SYNTHETIC_ARTICLE_CONTENTS.items():
        if not _matches_expected_json(contents.get(pageid), expected_content):
            violations.append(
                f"synthetic {relative_path} record {pageid} differs from the deterministic fixture"
            )


def _validate_small_synthetic_no_place_summaries(root: Path, violations: list[str]) -> None:
    relative_path = "data/wiki/article_summaries_no_place.json"
    summaries = _load_json(root / relative_path, violations)
    if not isinstance(summaries, dict):
        return
    for pageid, expected_summary in SMALL_SYNTHETIC_SUMMARIES_NO_PLACE.items():
        record = summaries.get(pageid)
        if not _matches_expected_json(record, {"summary": expected_summary}):
            actual = record.get("summary") if isinstance(record, dict) else record
            violations.append(
                f"synthetic {relative_path} record {pageid} summary={actual!r}; "
                f"expected {expected_summary!r}"
            )


def _validate_small_synthetic_contract(
    root: Path, wiki_pageids: set[str], violations: list[str]
) -> None:
    manifest = _load_json(root / "manifest.json", violations)
    if not isinstance(manifest, dict):
        return
    if manifest.get("data_source") != "synthetic":
        violations.append("small profile manifest data_source must be 'synthetic'")
        return

    expected_run_provenance = {
        "workflow": "reproduce_small",
        "model": SMALL_SYNTHETIC_MODEL,
        "seed": SMALL_SYNTHETIC_SEED,
        "temperature": SMALL_SYNTHETIC_TEMPERATURE,
        "classification_policy_version": classification_runner.CLASSIFICATION_POLICY_VERSION,
        "inputs": list(SMALL_SYNTHETIC_INPUTS),
        "outputs": list(SMALL_SYNTHETIC_OUTPUTS),
        "known_non_reproducible_components": [],
    }
    for field, expected in expected_run_provenance.items():
        actual = manifest.get(field)
        if not _matches_expected_json(actual, expected):
            violations.append(
                f"manifest.json synthetic run field {field}={actual!r}; expected {expected!r}"
            )

    for field in ("python_version", "project_version"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            violations.append(f"manifest.json {field} must be a non-empty string")

    created_at_utc = manifest.get("created_at_utc")
    if not isinstance(created_at_utc, str) or not created_at_utc.strip():
        violations.append("manifest.json created_at_utc must be an ISO-8601 UTC timestamp")
    else:
        timestamp = (
            f"{created_at_utc[:-1]}+00:00" if created_at_utc.endswith("Z") else created_at_utc
        )
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            violations.append("manifest.json created_at_utc must be an ISO-8601 UTC timestamp")
        else:
            if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timedelta(0):
                violations.append("manifest.json created_at_utc must be an ISO-8601 UTC timestamp")

    if wiki_pageids != SMALL_SYNTHETIC_PAGEIDS:
        missing = sorted(SMALL_SYNTHETIC_PAGEIDS - wiki_pageids)
        extra = sorted(wiki_pageids - SMALL_SYNTHETIC_PAGEIDS)
        violations.append(
            f"synthetic wiki pageids must be exactly {sorted(SMALL_SYNTHETIC_PAGEIDS)}: "
            f"missing={missing} extra={extra}"
        )

    _validate_small_synthetic_inputs(root, violations)
    _validate_small_synthetic_contents(root, violations)
    _validate_small_synthetic_summaries(root, violations)
    _validate_small_synthetic_no_place_summaries(root, violations)

    expected_counts = manifest.get("expected_counts")
    if isinstance(expected_counts, dict):
        unexpected_counts = sorted(set(expected_counts) - set(SMALL_SYNTHETIC_COUNTS))
        if unexpected_counts:
            violations.append(
                f"manifest.json expected_counts has unexpected keys: {unexpected_counts}"
            )
    for name, expected in SMALL_SYNTHETIC_COUNTS.items():
        actual = expected_counts.get(name) if isinstance(expected_counts, dict) else None
        if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected:
            violations.append(
                f"synthetic manifest expected_counts.{name}={actual!r}; expected {expected}"
            )

    run_dir = root / "data/classification/runs/small"
    for task, text_source in CLASSIFICATION_RUNS:
        stem = f"{task}_{text_source}"
        predictions_path = run_dir / f"{stem}_predictions.json"
        predictions = _load_json(predictions_path, violations)
        if isinstance(predictions, dict):
            prediction_pageids = {str(pageid) for pageid in predictions}
            if prediction_pageids != SMALL_SYNTHETIC_PAGEIDS:
                missing = sorted(SMALL_SYNTHETIC_PAGEIDS - prediction_pageids)
                extra = sorted(prediction_pageids - SMALL_SYNTHETIC_PAGEIDS)
                violations.append(
                    f"synthetic {stem} prediction pageids must be exactly "
                    f"{sorted(SMALL_SYNTHETIC_PAGEIDS)}: missing={missing} extra={extra}"
                )
            for pageid, expected_record in SMALL_SYNTHETIC_EXPECTED_PREDICTIONS[stem].items():
                record = predictions.get(pageid)
                if not isinstance(record, dict):
                    continue
                expected_fields = {
                    **expected_record,
                    "metadata": _expected_small_synthetic_metadata(task, pageid),
                }
                differing_fields = sorted(
                    field
                    for field in record.keys() | expected_fields.keys()
                    if (
                        field not in record
                        or field not in expected_fields
                        or not _matches_expected_json(record[field], expected_fields[field])
                    )
                )
                if differing_fields:
                    violations.append(
                        f"synthetic {stem} prediction {pageid} differs from deterministic "
                        f"expected values for fields {differing_fields}"
                    )

    for filename, expected_metrics in SMALL_SYNTHETIC_KNOWN_METRICS.items():
        metrics = _load_json(run_dir / f"{filename}.json", violations)
        if not isinstance(metrics, dict):
            continue
        if not _matches_expected_json(metrics, expected_metrics):
            differing_fields = sorted(
                field
                for field in metrics.keys() | expected_metrics.keys()
                if (
                    field not in metrics
                    or field not in expected_metrics
                    or not _matches_expected_json(metrics[field], expected_metrics[field])
                )
            )
            violations.append(
                f"synthetic {filename} metrics differ from deterministic expected values "
                f"for fields {differing_fields}"
            )


def _validate_small_artifacts(root: Path) -> list[str]:
    violations = _required_files(root, SMALL_REQUIRED_FILES)
    if violations:
        return violations
    _validate_manifest_hashes(root, violations)
    wiki_pageids = _validate_wiki_inputs(root, violations)
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
    _validate_small_synthetic_contract(root, wiki_pageids, violations)
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
