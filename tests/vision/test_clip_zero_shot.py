from pathlib import Path

import numpy as np
import pandas as pd

from georeset.vision.clip_zero_shot import (
    build_corine_zero_shot_prompts,
    predict_zero_shot,
    run_zero_shot_evaluation,
)


def test_build_corine_zero_shot_prompts_keeps_eval_label_order() -> None:
    prompts = build_corine_zero_shot_prompts(["31", "22"])

    assert list(prompts) == ["31", "22"]
    assert prompts["31"][0] == "a satellite image of forests"
    assert "permanent crops" in prompts["22"][0]


def test_predict_zero_shot_uses_cosine_similarity() -> None:
    image_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_embeddings = {
        "31": np.array([0.9, 0.1], dtype=np.float32),
        "22": np.array([0.1, 0.9], dtype=np.float32),
    }

    predictions = predict_zero_shot(image_embeddings, text_embeddings)

    assert predictions.tolist() == ["31", "22"]


def test_run_zero_shot_evaluation_writes_metrics_and_predictions(tmp_path: Path) -> None:
    splits_path = tmp_path / "splits.csv"
    embeddings_path = tmp_path / "embeddings.npz"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        [
            {"pageid": "1", "split": "eval_strict", "tier": "eval_strict", "label": "31"},
            {"pageid": "2", "split": "eval_strict", "tier": "eval_strict", "label": "22"},
        ]
    ).to_csv(splits_path, index=False)
    np.savez(
        embeddings_path,
        pageids=np.array(["1", "2"]),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    run_zero_shot_evaluation(
        splits_path=splits_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        text_encoder=lambda prompts: np.array(
            [[1.0, 0.0] if "forests" in prompt else [0.0, 1.0] for prompt in prompts],
            dtype=np.float32,
        ),
    )

    metrics = pd.read_csv(output_dir / "zero_shot_clip_metrics.csv")
    predictions = pd.read_csv(output_dir / "zero_shot_clip_predictions.csv")
    assert metrics.loc[0, "baseline"] == "zero_shot_clip"
    assert metrics.loc[0, "accuracy"] == 1.0
    assert predictions["prediction"].tolist() == [31, 22]
