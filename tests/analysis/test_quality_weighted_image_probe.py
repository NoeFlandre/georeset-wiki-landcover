from pathlib import Path

import numpy as np
import pandas as pd

from georeset.cli.analysis.evaluate_image_probe_training_policy_controls import evaluate_controls
from georeset.cli.analysis.run_quality_weighted_image_probe import run_probe


def _write_probe_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    splits_path = tmp_path / "splits.csv"
    weights_path = tmp_path / "weights.csv"
    embeddings_path = tmp_path / "embeddings_clip_base_window_0320m.npz"
    rows = [
        {"pageid": "1", "split": "train", "tier": "all", "label": "a"},
        {"pageid": "2", "split": "train", "tier": "all", "label": "a"},
        {"pageid": "3", "split": "train", "tier": "all", "label": "b"},
        {"pageid": "4", "split": "train", "tier": "all", "label": "b"},
        {"pageid": "1", "split": "train", "tier": "quality_spatial", "label": "a"},
        {"pageid": "2", "split": "train", "tier": "quality_spatial", "label": "a"},
        {"pageid": "3", "split": "train", "tier": "quality_spatial", "label": "b"},
        {"pageid": "4", "split": "train", "tier": "quality_spatial", "label": "b"},
        {"pageid": "1", "split": "train", "tier": "text_spatial_agreement", "label": "a"},
        {"pageid": "3", "split": "train", "tier": "text_spatial_agreement", "label": "b"},
        {"pageid": "5", "split": "eval_strict", "tier": "eval_strict", "label": "a"},
        {"pageid": "6", "split": "eval_strict", "tier": "eval_strict", "label": "b"},
    ]
    for row in rows:
        row.update(
            {
                "relevance_component": 1.0,
                "uncertainty_component": 1.0,
                "spatial_component": 1.0,
                "agreement_component": 1.0,
                "weight_raw": 1.0,
                "weight_class_balanced": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(splits_path, index=False)
    pd.DataFrame(
        [
            {
                "pageid": str(index),
                "label": "a" if index in {1, 2, 5} else "b",
                "relevance_component": 1.0,
                "uncertainty_component": 1.0,
                "spatial_component": 1.0,
                "agreement_component": 1.0,
                "weight_raw": 1.0,
                "weight_class_balanced": 1.0,
            }
            for index in range(1, 7)
        ]
    ).to_csv(weights_path, index=False)
    np.savez(
        embeddings_path,
        pageids=np.array(["1", "2", "3", "4", "5", "6"]),
        embeddings=np.array(
            [[2, 0], [1.8, 0.1], [0, 2], [0.1, 1.8], [2.1, 0], [0, 2.1]],
            dtype=np.float32,
        ),
        encoder_name=np.asarray("clip_base"),
        model_name=np.asarray("openai/clip-vit-base-patch32"),
        window_m=np.asarray(320),
    )
    return splits_path, weights_path, embeddings_path


def test_run_quality_weighted_image_probe_writes_required_outputs(tmp_path: Path) -> None:
    splits_path, weights_path, embeddings_path = _write_probe_fixture(tmp_path)
    output_dir = tmp_path / "out"

    run_probe(
        splits_path=splits_path,
        weights_path=weights_path,
        embeddings_paths=[embeddings_path],
        output_dir=output_dir,
        seed=5,
        epochs=20,
        learning_rate=0.2,
        n_bootstrap=3,
    )

    assert (output_dir / "weighted_probe_metrics.csv").exists()
    assert (output_dir / "weighted_probe_predictions.csv").exists()
    assert (output_dir / "per_class_metrics.csv").exists()
    assert (output_dir / "bootstrap_confidence_intervals.csv").exists()
    assert (output_dir / "confusion_matrices.json").exists()
    assert (output_dir / "run_manifest.json").exists()
    metrics = pd.read_csv(output_dir / "weighted_probe_metrics.csv")
    assert {"balanced_accuracy_supported", "balanced_accuracy_allowed"}.issubset(metrics.columns)


def test_evaluate_image_probe_training_policy_controls_writes_required_outputs(
    tmp_path: Path,
) -> None:
    splits_path, _, embeddings_path = _write_probe_fixture(tmp_path)
    output_dir = tmp_path / "controls"

    evaluate_controls(
        splits_path=splits_path,
        embeddings_paths=[embeddings_path],
        output_dir=output_dir,
        n_draws=2,
        seed=5,
        epochs=20,
        learning_rate=0.2,
        l2=1e-4,
    )

    assert (output_dir / "image_probe_random_training_controls.csv").exists()
    assert (output_dir / "image_probe_random_training_controls.md").exists()
    assert (output_dir / "control_manifest.json").exists()
