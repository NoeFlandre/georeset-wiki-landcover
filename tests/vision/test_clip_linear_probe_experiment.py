from pathlib import Path

import numpy as np
import pandas as pd

from georeset.cli.analysis.run_clip_linear_probe_experiment import main


def test_clip_linear_probe_experiment_trains_tiers_from_cached_embeddings(
    tmp_path: Path,
) -> None:
    splits_path = tmp_path / "splits.csv"
    embeddings_path = tmp_path / "embeddings.npz"
    output_dir = tmp_path / "out"
    splits = pd.DataFrame(
        [
            {"pageid": "1", "split": "train", "tier": "all", "label": "31"},
            {"pageid": "2", "split": "train", "tier": "all", "label": "31"},
            {"pageid": "3", "split": "train", "tier": "all", "label": "22"},
            {"pageid": "4", "split": "train", "tier": "all", "label": "22"},
            {"pageid": "1", "split": "train", "tier": "text_spatial_agreement", "label": "31"},
            {"pageid": "2", "split": "train", "tier": "text_spatial_agreement", "label": "31"},
            {"pageid": "3", "split": "train", "tier": "text_spatial_agreement", "label": "22"},
            {"pageid": "4", "split": "train", "tier": "text_spatial_agreement", "label": "22"},
            {"pageid": "5", "split": "eval_strict", "tier": "eval_strict", "label": "31"},
            {"pageid": "6", "split": "eval_strict", "tier": "eval_strict", "label": "22"},
        ]
    )
    splits.to_csv(splits_path, index=False)
    np.savez(
        embeddings_path,
        pageids=np.array(["1", "2", "3", "4", "5", "6"]),
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.0],
                [0.0, 0.8],
            ],
            dtype=np.float32,
        ),
    )

    main(
        [
            "--splits-path",
            str(splits_path),
            "--embeddings-path",
            str(embeddings_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "250",
            "--learning-rate",
            "0.2",
        ]
    )

    metrics = pd.read_csv(output_dir / "linear_probe_metrics.csv")
    assert set(metrics["tier"]) == {"all", "text_spatial_agreement"}
    assert metrics["accuracy"].min() == 1.0
    assert (output_dir / "linear_probe_predictions.csv").exists()
    assert (output_dir / "summary.md").exists()
