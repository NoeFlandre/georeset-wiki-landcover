import json
from pathlib import Path

import pandas as pd

from georeset_wiki_landcover.cli.analysis.evaluate_evidence_card_experiment import main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _prediction(target: object, prediction: object, status: str = "ok") -> dict[str, object]:
    return {
        "target": target,
        "prediction": prediction,
        "parse_status": status,
        "metadata": {"model": "Qwen3.6-27B-Q4_0.gguf"},
    }


def _build_experiment(path: Path, sources: list[str]) -> None:
    for source in sources:
        corine_prediction = "31" if "shuffled" not in source else "21"
        osm_prediction = ["wood"] if "shuffled" not in source else ["water"]
        _write_json(
            path / f"corine_level2_{source}_predictions.json",
            {
                "1": _prediction("31", corine_prediction),
                "2": _prediction("21", "21"),
            },
        )
        _write_json(
            path / f"osm_{source}_predictions.json",
            {
                "1": _prediction(["wood"], osm_prediction),
                "2": _prediction(["water"], ["water"]),
            },
        )


def test_evaluate_evidence_card_experiment_writes_comparison_outputs(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    previous_dir = tmp_path / "previous"
    landuse_dir = tmp_path / "landuse"
    output_dir = tmp_path / "out"
    frozen_sentinel = previous_dir / "sentinel.txt"
    previous_dir.mkdir()
    frozen_sentinel.write_text("unchanged", encoding="utf-8")

    _build_experiment(
        evidence_dir,
        [
            "evidence_card",
            "evidence_card_shuffled",
            "content_with_evidence_card",
            "content_with_evidence_card_shuffled",
        ],
    )
    _build_experiment(previous_dir, ["summary", "summary_no_place", "content"])
    _build_experiment(landuse_dir, ["landuse_evidence_summary"])

    quality_scores = pd.DataFrame(
        [
            {
                "pageid": "1",
                "landcover_relevance": "high",
                "point_label_share_250m": 0.9,
                "quality_bin": "quality_high",
                "recommended_use": "use_for_training",
            },
            {
                "pageid": "2",
                "landcover_relevance": "none",
                "point_label_share_250m": 0.2,
                "quality_bin": "quality_low",
                "recommended_use": "exclude",
            },
        ]
    )
    quality_path = tmp_path / "quality_scores.csv"
    quality_scores.to_csv(quality_path, index=False)

    main(
        [
            "--evidence-card-experiment-dir",
            str(evidence_dir),
            "--previous-qwen-experiment-dir",
            str(previous_dir),
            "--landuse-evidence-experiment-dir",
            str(landuse_dir),
            "--quality-scores-path",
            str(quality_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    for name in [
        "evidence_card_vs_previous_sources.csv",
        "evidence_card_quality_subsets.csv",
        "evidence_card_shuffled_deltas.csv",
        "evidence_card_per_class_corine.csv",
        "evidence_card_osm_metrics.csv",
        "manifest.json",
        "summary.md",
    ]:
        assert (output_dir / name).exists()

    overview = pd.read_csv(output_dir / "evidence_card_vs_previous_sources.csv")
    assert {"evidence_card", "landuse_evidence_summary", "content"}.issubset(
        set(overview["text_source"])
    )
    quality = pd.read_csv(output_dir / "evidence_card_quality_subsets.csv")
    assert "quality_high_or_very_high" in set(quality["subset"])
    deltas = pd.read_csv(output_dir / "evidence_card_shuffled_deltas.csv")
    assert "content_with_evidence_card" in set(deltas["text_source"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_summarization_rerun"] is True
    assert manifest["deterministic_card_version"] == 1
    assert frozen_sentinel.read_text(encoding="utf-8") == "unchanged"


def test_evaluate_evidence_card_experiment_handles_missing_quality_scores(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    previous_dir = tmp_path / "previous"
    landuse_dir = tmp_path / "landuse"
    output_dir = tmp_path / "out"

    _build_experiment(evidence_dir, ["evidence_card", "evidence_card_shuffled"])
    _build_experiment(previous_dir, ["content"])
    _build_experiment(landuse_dir, ["landuse_evidence_summary"])

    main(
        [
            "--evidence-card-experiment-dir",
            str(evidence_dir),
            "--previous-qwen-experiment-dir",
            str(previous_dir),
            "--landuse-evidence-experiment-dir",
            str(landuse_dir),
            "--quality-scores-path",
            str(tmp_path / "missing_quality_scores.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )

    overview = pd.read_csv(output_dir / "evidence_card_quality_subsets.csv")
    assert "all" in set(overview["subset"])
    assert "quality_high_or_very_high" not in set(overview["subset"])
