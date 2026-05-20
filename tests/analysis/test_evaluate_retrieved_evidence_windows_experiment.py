import json
from pathlib import Path

import pandas as pd

from georeset_wiki_landcover.cli.analysis.evaluate_retrieved_evidence_windows_experiment import main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _prediction(target: object, prediction: object, model: str) -> dict[str, object]:
    return {
        "target": target,
        "prediction": prediction,
        "prediction_labels": [prediction] if isinstance(prediction, str) else prediction,
        "parse_status": "ok",
        "metadata": {"model": model},
    }


def _build_experiment(path: Path, model: str, sources: list[str]) -> None:
    for source in sources:
        q = source
        corine_second = "21" if "random" not in q and "shuffled" not in q else "31"
        _write_json(
            path / f"corine_level2_{source}_predictions.json",
            {
                "1": _prediction("31", "31", model),
                "2": _prediction("21", corine_second, model),
            },
        )
        osm_second = ["water"] if "random" not in q and "shuffled" not in q else ["wood"]
        _write_json(
            path / f"osm_{source}_predictions.json",
            {
                "1": _prediction(["wood"], ["wood"], model),
                "2": _prediction(["water"], osm_second, model),
            },
        )


def test_evaluate_retrieved_evidence_windows_experiment_writes_research_outputs(
    tmp_path: Path,
) -> None:
    qwen_retrieved = tmp_path / "qwen_retrieved"
    gemma_retrieved = tmp_path / "gemma_retrieved"
    qwen_previous = tmp_path / "qwen_previous"
    gemma_previous = tmp_path / "gemma_previous"
    output_dir = tmp_path / "out"
    sources = [
        "retrieved_evidence_windows",
        "retrieved_evidence_sentences_only",
        "random_sentence_windows",
        "retrieved_evidence_windows_no_place",
        "retrieved_evidence_windows_shuffled",
    ]
    _build_experiment(qwen_retrieved, "Qwen3.6-27B-Q4_0.gguf", sources)
    _build_experiment(gemma_retrieved, "gemma-4-31B-it-Q4_0.gguf", sources)
    _build_experiment(qwen_previous, "Qwen3.6-27B-Q4_0.gguf", ["content", "content_shuffled"])
    _build_experiment(gemma_previous, "gemma-4-31B-it-Q4_0.gguf", ["content", "content_shuffled"])
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
    quality_path = tmp_path / "quality.csv"
    quality_scores.to_csv(quality_path, index=False)

    main(
        [
            "--qwen-retrieved-experiment-dir",
            str(qwen_retrieved),
            "--gemma-retrieved-experiment-dir",
            str(gemma_retrieved),
            "--qwen-previous-experiment-dir",
            str(qwen_previous),
            "--gemma-previous-experiment-dir",
            str(gemma_previous),
            "--quality-scores-path",
            str(quality_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    for name in [
        "retrieved_evidence_windows_quality_subsets.csv",
        "retrieved_evidence_windows_shuffled_deltas.csv",
        "retrieved_evidence_windows_pairwise_deltas.csv",
        "retrieved_evidence_windows_model_agreement.csv",
        "retrieved_evidence_windows_per_class_corine.csv",
        "manifest.json",
        "summary.md",
    ]:
        assert (output_dir / name).exists()

    pairwise = pd.read_csv(output_dir / "retrieved_evidence_windows_pairwise_deltas.csv")
    assert {"content_minus_retrieved", "retrieved_minus_random"}.issubset(
        set(pairwise["comparison"])
    )
    agreement = pd.read_csv(output_dir / "retrieved_evidence_windows_model_agreement.csv")
    assert {"agreement_rate", "agreement_precision_vs_target"}.issubset(agreement.columns)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "retrieved_evidence_windows_comparison_v1"
