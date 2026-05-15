import json
from pathlib import Path

import pandas as pd

from georeset.cli.data.build_clip_label_splits import main


def test_build_clip_label_splits_cli_writes_csv(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.csv"
    qwen_path = tmp_path / "qwen.json"
    gemma_path = tmp_path / "gemma.json"
    output_path = tmp_path / "splits.csv"
    pd.DataFrame(
        [
            {
                "pageid": str(index),
                "title": f"Article {index}",
                "lat": 48.0,
                "lon": 7.0,
                "corine_label": "31" if index < 3 else "22",
                "quality_bin": "quality_high",
                "recommended_use": "use_for_training",
                "landcover_relevance": "high",
                "uncertainty": "low",
                "point_label_share_250m": 0.9,
                "dominant_matches_point_label_250m": True,
            }
            for index in range(6)
        ]
    ).to_csv(quality_path, index=False)
    predictions = {
        str(index): {
            "prediction": "31" if index < 3 else "22",
            "parse_status": "ok",
        }
        for index in range(6)
    }
    qwen_path.write_text(json.dumps(predictions), encoding="utf-8")
    gemma_path.write_text(json.dumps(predictions), encoding="utf-8")

    main(
        [
            "--quality-scores-path",
            str(quality_path),
            "--qwen-predictions-path",
            str(qwen_path),
            "--gemma-predictions-path",
            str(gemma_path),
            "--output-path",
            str(output_path),
            "--eval-per-class",
            "1",
            "--train-per-class",
            "1",
        ]
    )

    output = pd.read_csv(output_path, dtype={"pageid": str, "label": str})
    assert {"eval_strict", "train"} == set(output["split"])
    assert "text_spatial_agreement" in set(output["tier"])
