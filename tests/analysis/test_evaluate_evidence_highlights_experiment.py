import json
from pathlib import Path

import pandas as pd

from georeset_wiki_landcover.cli.analysis.evaluate_evidence_highlights_experiment import main


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


def test_evaluate_evidence_highlights_experiment_writes_comparison_outputs(tmp_path: Path):
    qwen_highlight_dir = tmp_path / "qwen_highlights"
    gemma_highlight_dir = tmp_path / "gemma_highlights"
    qwen_previous_dir = tmp_path / "qwen_previous"
    gemma_previous_dir = tmp_path / "gemma_previous"
    qwen_landuse_dir = tmp_path / "qwen_landuse"
    gemma_landuse_dir = tmp_path / "gemma_landuse"
    qwen_card_dir = tmp_path / "qwen_card"
    output_dir = tmp_path / "out"
    frozen_sentinel = qwen_previous_dir / "sentinel.txt"
    qwen_previous_dir.mkdir()
    frozen_sentinel.write_text("unchanged", encoding="utf-8")

    highlight_sources = [
        "content_with_evidence_highlights",
        "content_with_evidence_highlights_shuffled",
    ]
    _build_experiment(qwen_highlight_dir, highlight_sources)
    _build_experiment(gemma_highlight_dir, highlight_sources)
    _build_experiment(qwen_previous_dir, ["summary", "summary_no_place", "content"])
    _build_experiment(gemma_previous_dir, ["summary", "summary_no_place", "content"])
    _build_experiment(qwen_landuse_dir, ["landuse_evidence_summary"])
    _build_experiment(gemma_landuse_dir, ["landuse_evidence_summary"])
    _build_experiment(qwen_card_dir, ["content_with_evidence_card"])

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
            "--qwen-evidence-highlights-experiment-dir",
            str(qwen_highlight_dir),
            "--gemma-evidence-highlights-experiment-dir",
            str(gemma_highlight_dir),
            "--qwen-previous-experiment-dir",
            str(qwen_previous_dir),
            "--gemma-previous-experiment-dir",
            str(gemma_previous_dir),
            "--qwen-landuse-evidence-experiment-dir",
            str(qwen_landuse_dir),
            "--gemma-landuse-evidence-experiment-dir",
            str(gemma_landuse_dir),
            "--qwen-evidence-card-experiment-dir",
            str(qwen_card_dir),
            "--quality-scores-path",
            str(quality_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    for name in [
        "evidence_highlights_vs_previous_sources.csv",
        "evidence_highlights_quality_subsets.csv",
        "evidence_highlights_shuffled_deltas.csv",
        "evidence_highlights_per_class_corine.csv",
        "evidence_highlights_osm_metrics.csv",
        "manifest.json",
        "summary.md",
    ]:
        assert (output_dir / name).exists()

    overview = pd.read_csv(output_dir / "evidence_highlights_vs_previous_sources.csv")
    assert {"content_with_evidence_highlights", "landuse_evidence_summary", "content"}.issubset(
        set(overview["text_source"])
    )
    assert {"qwen", "gemma4_31b_it_q4_0"}.issubset(set(overview["model_key"]))
    quality = pd.read_csv(output_dir / "evidence_highlights_quality_subsets.csv")
    assert "quality_high_or_very_high" in set(quality["subset"])
    deltas = pd.read_csv(output_dir / "evidence_highlights_shuffled_deltas.csv")
    assert "content_with_evidence_highlights" in set(deltas["text_source"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_llm_highlight_generation"] is True
    assert manifest["deterministic_highlight_version"] == 1
    assert frozen_sentinel.read_text(encoding="utf-8") == "unchanged"


def test_evaluate_evidence_highlights_experiment_handles_missing_quality_scores(tmp_path: Path):
    qwen_highlight_dir = tmp_path / "qwen_highlights"
    output_dir = tmp_path / "out"
    _build_experiment(
        qwen_highlight_dir,
        ["content_with_evidence_highlights", "content_with_evidence_highlights_shuffled"],
    )

    main(
        [
            "--qwen-evidence-highlights-experiment-dir",
            str(qwen_highlight_dir),
            "--quality-scores-path",
            str(tmp_path / "missing_quality_scores.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )

    overview = pd.read_csv(output_dir / "evidence_highlights_quality_subsets.csv")
    assert "all" in set(overview["subset"])
    assert "quality_high_or_very_high" not in set(overview["subset"])
