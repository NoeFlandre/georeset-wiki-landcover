"""Build a tiny synthetic reproduction package without private data or an LLM."""

from __future__ import annotations

import argparse
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import box

from georeset_wiki_landcover.classification import runner as classification_runner
from georeset_wiki_landcover.classification.types import PredictionResult
from georeset_wiki_landcover.utils.json_io import write_json_atomic

try:
    from scripts.validate_artifacts import sha256_file, validate_artifacts
except ModuleNotFoundError:
    from validate_artifacts import sha256_file, validate_artifacts

SYNTHETIC_MODEL_NAME = "synthetic-deterministic-classifier"
SMALL_RUN_DIR = Path("data/classification/runs/small")


class DeterministicClassifier:
    """Small local classifier used only by the reproducibility smoke workflow."""

    def __init__(self, model_path: str | None, seed: int, temperature: float) -> None:
        self.model_path = model_path or SYNTHETIC_MODEL_NAME
        self.seed = seed
        self.temperature = temperature

    def _metadata(self, task: str, text_source: str, allowed_labels: list[str]) -> dict[str, Any]:
        return {
            "task": task,
            "text_source": text_source,
            "model": self.model_path,
            "model_repo_id": None,
            "seed": self.seed,
            "temperature": self.temperature,
            "allowed_labels": sorted(allowed_labels),
            "prompt": "Synthetic deterministic classifier; no prompt sent to an LLM.",
            "system_prompt": "Synthetic deterministic classifier.",
            "attempt_count": 1,
        }

    def classify_single_label(
        self,
        text: str,
        allowed_labels: list[str],
        label_descriptions: dict[str, str],
        task: str,
        text_source: str,
    ) -> PredictionResult:
        del label_descriptions
        prediction = "31" if "forest" in text.lower() else "21"
        if prediction not in allowed_labels:
            prediction = allowed_labels[0]
        return {
            "prediction": prediction,
            "prediction_labels": [prediction],
            "parse_status": "ok",
            "raw_response": f'{{"label": "{prediction}"}}',
            "error": None,
            "metadata": self._metadata(task, text_source, allowed_labels),
        }

    def classify_multilabel(
        self,
        text: str,
        allowed_labels: list[str],
        task: str,
        text_source: str,
    ) -> PredictionResult:
        prediction = ["wood"] if "forest" in text.lower() else ["meadow"]
        prediction = [label for label in prediction if label in allowed_labels] or [
            allowed_labels[0]
        ]
        return {
            "prediction": prediction,
            "prediction_labels": prediction,
            "parse_status": "ok",
            "raw_response": f'{{"labels": {prediction!r}}}',
            "error": None,
            "metadata": self._metadata(task, text_source, allowed_labels),
        }


def _project_version() -> str:
    try:
        return version("georeset-wiki-landcover")
    except PackageNotFoundError:
        return "not-installed"


def _clean_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    unsafe_paths = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if resolved in unsafe_paths:
        raise ValueError(f"refusing to clean unsafe output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _write_synthetic_inputs(output_dir: Path) -> dict[str, Path]:
    wiki_dir = output_dir / "data/wiki"
    corine_dir = output_dir / "data/corine"
    osm_dir = output_dir / "data/osm"
    for directory in (wiki_dir, corine_dir, osm_dir):
        directory.mkdir(parents=True, exist_ok=True)

    articles = [
        {
            "pageid": 100,
            "title": "Synthetic Forest",
            "lat": 0.5,
            "lon": 0.5,
            "url": "https://example.invalid/wiki/Synthetic_Forest",
        },
        {
            "pageid": 200,
            "title": "Synthetic Meadow",
            "lat": 0.5,
            "lon": 2.5,
            "url": "https://example.invalid/wiki/Synthetic_Meadow",
        },
    ]
    contents = {
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
    summaries = {
        "100": {"summary": "Synthetic forest land cover with trees."},
        "200": {"summary": "Synthetic meadow land cover with grass."},
    }
    no_place = {
        "100": {"summary": "A forest-like area with trees."},
        "200": {"summary": "A meadow-like area with grass."},
    }

    wiki_articles_path = wiki_dir / "wiki_articles.json"
    contents_path = wiki_dir / "article_contents.json"
    summaries_path = wiki_dir / "article_summaries.json"
    no_place_path = wiki_dir / "article_summaries_no_place.json"
    write_json_atomic(wiki_articles_path, articles, indent=2)
    write_json_atomic(contents_path, contents, indent=2)
    write_json_atomic(summaries_path, summaries, indent=2)
    write_json_atomic(no_place_path, no_place, indent=2)

    corine_path = corine_dir / "synthetic_corine.geojson"
    corine = gpd.GeoDataFrame(
        {"ID": [1, 2], "code_18": ["311", "211"]},
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )
    corine.to_file(corine_path, driver="GeoJSON")

    osm_path = osm_dir / "osm_project_polygons.geojson"
    osm = gpd.GeoDataFrame(
        {
            "osm_id": ["synthetic/wood", "synthetic/meadow"],
            "landuse": [None, "meadow"],
            "natural": ["wood", None],
        },
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )
    osm.to_file(osm_path, driver="GeoJSON")

    return {
        "wiki_articles": wiki_articles_path,
        "article_contents": contents_path,
        "article_summaries": summaries_path,
        "article_summaries_no_place": no_place_path,
        "corine_polygons": corine_path,
        "osm_polygons": osm_path,
    }


def _run_classification(output_dir: Path, paths: dict[str, Path]) -> None:
    run_dir = output_dir / SMALL_RUN_DIR
    common_args = [
        "--text-source",
        "summary",
        "--wiki-articles-path",
        str(paths["wiki_articles"]),
        "--article-contents-path",
        str(paths["article_contents"]),
        "--article-summaries-path",
        str(paths["article_summaries"]),
        "--article-summaries-no-place-path",
        str(paths["article_summaries_no_place"]),
        "--output-dir",
        str(run_dir),
        "--model-path",
        SYNTHETIC_MODEL_NAME,
        "--seed",
        "42",
        "--temperature",
        "0.0",
        "--limit",
        "2",
    ]

    def classifier_factory(
        model_path: str | None, model_repo_id: str | None, seed: int, temperature: float
    ) -> DeterministicClassifier:
        del model_repo_id
        return DeterministicClassifier(model_path, seed, temperature)

    classification_runner.main(
        [
            "--task",
            "corine_level2",
            "--corine-polygons-path",
            str(paths["corine_polygons"]),
            *common_args,
        ],
        classifier_factory=classifier_factory,
    )
    classification_runner.main(
        [
            "--task",
            "osm",
            "--osm-polygons-path",
            str(paths["osm_polygons"]),
            *common_args,
        ],
        classifier_factory=classifier_factory,
    )


def _manifest(output_dir: Path) -> dict[str, Any]:
    artifact_paths = [
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
    ]
    return {
        "workflow": "reproduce_small",
        "mode": "small",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python_version": platform.python_version(),
        "project_version": _project_version(),
        "classification_policy_version": classification_runner.CLASSIFICATION_POLICY_VERSION,
        "data_source": "synthetic",
        "model": SYNTHETIC_MODEL_NAME,
        "seed": 42,
        "temperature": 0.0,
        "expected_counts": {
            "wiki_articles": 2,
            "corine_level2_summary_predictions": 2,
            "osm_summary_predictions": 2,
        },
        "inputs": artifact_paths[:6],
        "outputs": artifact_paths[6:],
        "artifact_sha256": {
            relative_path: sha256_file(output_dir / relative_path)
            for relative_path in artifact_paths
        },
        "known_non_reproducible_components": [],
    }


def run_small_reproduction(output_dir: Path | str, *, clean: bool = False) -> dict[str, Any]:
    """Generate tiny synthetic inputs, run deterministic classification, and validate outputs."""
    output_path = Path(output_dir)
    if clean:
        _clean_output_dir(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = _write_synthetic_inputs(output_path)
    _run_classification(output_path, paths)
    manifest = _manifest(output_path)
    write_json_atomic(output_path / "manifest.json", manifest, indent=2)

    violations = validate_artifacts(output_path, profile="small")
    if violations:
        raise RuntimeError("small reproduction validation failed: " + "; ".join(violations))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/reproducibility/small"),
        help="Directory where synthetic inputs and outputs will be written.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove output dir before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_small_reproduction(output_dir=args.output_dir, clean=args.clean)
    print(f"Small reproduction complete: {args.output_dir}")
    print(f"Manifested artifacts: {len(manifest['artifact_sha256'])}")


if __name__ == "__main__":
    main()
