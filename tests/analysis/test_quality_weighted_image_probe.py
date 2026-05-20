from pathlib import Path

import numpy as np
import pandas as pd

from georeset.cli.analysis.evaluate_image_probe_training_policy_controls import (
    evaluate_controls,
    image_probe_control_manifest_path,
    image_probe_random_controls_markdown_path,
    image_probe_random_controls_path,
)
from georeset.cli.analysis.run_quality_weighted_image_probe import (
    bootstrap_confidence_intervals_path,
    confusion_matrices_path,
    per_class_metrics_path,
    run_manifest_path,
    run_probe,
    weighted_probe_metrics_path,
    weighted_probe_predictions_path,
    weighted_probe_summary_path,
)
from georeset.cli.analysis.run_quality_weighted_image_zero_shot import (
    run_zero_shot_image_probe,
    zero_shot_image_probe_metrics_path,
    zero_shot_image_probe_predictions_path,
    zero_shot_image_probe_summary_path,
)


def _write_probe_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    splits_path = tmp_path / "splits.csv"
    weights_path = tmp_path / "weights.csv"
    embeddings_path = tmp_path / "embeddings_clip_base_window_0320m.npz"
    rows = [
        {"pageid": "1", "split": "train", "tier": "all", "label": "31"},
        {"pageid": "2", "split": "train", "tier": "all", "label": "31"},
        {"pageid": "3", "split": "train", "tier": "all", "label": "22"},
        {"pageid": "4", "split": "train", "tier": "all", "label": "22"},
        {"pageid": "1", "split": "train", "tier": "quality_spatial", "label": "31"},
        {"pageid": "2", "split": "train", "tier": "quality_spatial", "label": "31"},
        {"pageid": "3", "split": "train", "tier": "quality_spatial", "label": "22"},
        {"pageid": "4", "split": "train", "tier": "quality_spatial", "label": "22"},
        {"pageid": "1", "split": "train", "tier": "text_spatial_agreement", "label": "31"},
        {"pageid": "3", "split": "train", "tier": "text_spatial_agreement", "label": "22"},
        {"pageid": "5", "split": "eval_strict", "tier": "eval_strict", "label": "31"},
        {"pageid": "6", "split": "eval_strict", "tier": "eval_strict", "label": "22"},
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
                "label": "31" if index in {1, 2, 5} else "22",
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


def _write_probe_fixture_with_repeated_split(tmp_path: Path) -> tuple[Path, Path, Path]:
    splits_path, weights_path, embeddings_path = _write_probe_fixture(tmp_path)
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    repeated = pd.DataFrame(
        [
            {
                "pageid": "1",
                "split": "repeated_eval_seed_1",
                "tier": "eval_repeated",
                "label": "31",
            },
            {
                "pageid": "6",
                "split": "repeated_eval_seed_1",
                "tier": "eval_repeated",
                "label": "22",
            },
        ]
    )
    for column in [
        "relevance_component",
        "uncertainty_component",
        "spatial_component",
        "agreement_component",
        "weight_raw",
        "weight_class_balanced",
    ]:
        repeated[column] = 1.0
    pd.concat([splits, repeated], ignore_index=True).to_csv(splits_path, index=False)
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


def test_run_quality_weighted_image_probe_excludes_each_eval_split_from_training(
    tmp_path: Path,
) -> None:
    splits_path, weights_path, embeddings_path = _write_probe_fixture_with_repeated_split(tmp_path)
    output_dir = tmp_path / "out"

    run_probe(
        splits_path=splits_path,
        weights_path=weights_path,
        embeddings_paths=[embeddings_path],
        output_dir=output_dir,
        seed=5,
        epochs=20,
        learning_rate=0.2,
        n_bootstrap=0,
    )

    metrics = pd.read_csv(output_dir / "weighted_probe_metrics.csv")
    repeated_all = metrics[
        (metrics["split"] == "repeated_eval_seed_1")
        & (metrics["policy"] == "all_unweighted")
        & (metrics["l2"] == 0.0001)
    ].iloc[0]
    assert repeated_all["n_train"] == 3


def test_run_quality_weighted_image_probe_records_l2_selection_as_exploratory(
    tmp_path: Path,
) -> None:
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
        n_bootstrap=0,
    )

    manifest = (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    assert '"model_selection": "exploratory_eval_grid_no_validation_split"' in manifest


def test_quality_weighted_image_probe_output_paths_use_expected_names(tmp_path: Path) -> None:
    assert weighted_probe_metrics_path(tmp_path) == tmp_path / "weighted_probe_metrics.csv"
    assert weighted_probe_predictions_path(tmp_path) == tmp_path / "weighted_probe_predictions.csv"
    assert per_class_metrics_path(tmp_path) == tmp_path / "per_class_metrics.csv"
    assert (
        bootstrap_confidence_intervals_path(tmp_path)
        == tmp_path / "bootstrap_confidence_intervals.csv"
    )
    assert confusion_matrices_path(tmp_path) == tmp_path / "confusion_matrices.json"
    assert run_manifest_path(tmp_path) == tmp_path / "run_manifest.json"
    assert weighted_probe_summary_path(tmp_path) == tmp_path / "summary.md"


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


def test_image_probe_training_policy_control_output_paths_use_expected_names(
    tmp_path: Path,
) -> None:
    assert (
        image_probe_random_controls_path(tmp_path)
        == tmp_path / "image_probe_random_training_controls.csv"
    )
    assert (
        image_probe_random_controls_markdown_path(tmp_path)
        == tmp_path / "image_probe_random_training_controls.md"
    )
    assert image_probe_control_manifest_path(tmp_path) == tmp_path / "control_manifest.json"


def test_quality_weighted_zero_shot_output_paths_use_expected_names(tmp_path: Path) -> None:
    assert (
        zero_shot_image_probe_metrics_path(tmp_path)
        == tmp_path / "zero_shot_image_probe_metrics.csv"
    )
    assert (
        zero_shot_image_probe_predictions_path(tmp_path)
        == tmp_path / "zero_shot_image_probe_predictions.csv"
    )
    assert (
        zero_shot_image_probe_summary_path(tmp_path)
        == tmp_path / "zero_shot_image_probe_summary.md"
    )


def test_run_quality_weighted_image_zero_shot_writes_window_aware_outputs(
    tmp_path: Path,
) -> None:
    splits_path, _, embeddings_path = _write_probe_fixture(tmp_path)
    output_dir = tmp_path / "zero-shot"

    def text_encoder_factory(model_name: str, device: str):
        assert model_name == "openai/clip-vit-base-patch32"
        assert device == "cpu"

        def encode(prompts: list[str]) -> np.ndarray:
            return np.array(
                [[1.0, 0.0] if "forests" in prompt else [0.0, 1.0] for prompt in prompts],
                dtype=np.float32,
            )

        return encode

    run_zero_shot_image_probe(
        splits_path=splits_path,
        embeddings_paths=[embeddings_path],
        output_dir=output_dir,
        device="cpu",
        text_encoder_factory=text_encoder_factory,
    )

    metrics = pd.read_csv(output_dir / "zero_shot_image_probe_metrics.csv")
    predictions = pd.read_csv(output_dir / "zero_shot_image_probe_predictions.csv")
    assert metrics.loc[0, "baseline"] == "zero_shot_clip"
    assert metrics.loc[0, "encoder"] == "clip_base"
    assert metrics.loc[0, "window_m"] == 320
    assert {"balanced_accuracy_supported", "macro_f1_supported"}.issubset(metrics.columns)
    assert predictions["window_m"].tolist() == [320, 320]


def test_run_quality_weighted_image_zero_shot_uses_full_split_label_universe(
    tmp_path: Path,
) -> None:
    splits_path, _, embeddings_path = _write_probe_fixture(tmp_path)
    splits = pd.read_csv(splits_path, dtype={"pageid": str, "label": str})
    splits.loc[len(splits)] = {
        **splits.iloc[0].to_dict(),
        "pageid": "7",
        "split": "train",
        "tier": "all",
        "label": "21",
    }
    splits.to_csv(splits_path, index=False)
    output_dir = tmp_path / "zero-shot"
    prompt_count = 0

    def text_encoder_factory(model_name: str, device: str):
        del model_name, device

        def encode(prompts: list[str]) -> np.ndarray:
            nonlocal prompt_count
            prompt_count += len(prompts)
            return np.ones((len(prompts), 2), dtype=np.float32)

        return encode

    run_zero_shot_image_probe(
        splits_path=splits_path,
        embeddings_paths=[embeddings_path],
        output_dir=output_dir,
        device="cpu",
        text_encoder_factory=text_encoder_factory,
    )

    metrics = pd.read_csv(output_dir / "zero_shot_image_probe_metrics.csv")
    assert metrics.loc[0, "n_labels"] == 3
    assert prompt_count == 9
