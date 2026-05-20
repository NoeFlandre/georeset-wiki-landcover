from pathlib import Path

import pandas as pd
import pytest

from georeset_wiki_landcover.vision.clip_weak_labels import build_clip_label_splits


def _write_predictions(path: Path, rows: dict[str, str]) -> None:
    path.write_text(
        "{"
        + ",".join(
            f'"{pageid}":{{"prediction":"{prediction}","parse_status":"ok"}}'
            for pageid, prediction in rows.items()
        )
        + "}",
        encoding="utf-8",
    )


def test_build_clip_label_splits_creates_fixed_eval_and_train_tiers(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.csv"
    qwen_path = tmp_path / "qwen.json"
    gemma_path = tmp_path / "gemma.json"
    rows = []
    for label in ["21", "22", "31"]:
        for index in range(6):
            pageid = f"{label}{index}"
            rows.append(
                {
                    "pageid": pageid,
                    "title": f"Article {pageid}",
                    "lat": 48.0 + index / 100,
                    "lon": 7.0 + index / 100,
                    "corine_label": label,
                    "quality_bin": "quality_high" if index < 5 else "quality_medium",
                    "recommended_use": "use_for_training",
                    "landcover_relevance": "high" if index != 5 else "low",
                    "uncertainty": "low",
                    "point_label_share_250m": 0.9 if index != 4 else 0.6,
                    "dominant_matches_point_label_250m": index != 4,
                }
            )
    pd.DataFrame(rows).to_csv(quality_path, index=False)
    agree = {row["pageid"]: row["corine_label"] for row in rows if not row["pageid"].endswith("3")}
    disagree = {row["pageid"]: "51" for row in rows if row["pageid"].endswith("3")}
    _write_predictions(qwen_path, {**agree, **disagree})
    _write_predictions(gemma_path, agree | disagree)

    splits = build_clip_label_splits(
        quality_scores_path=quality_path,
        qwen_predictions_path=qwen_path,
        gemma_predictions_path=gemma_path,
        seed=11,
        eval_per_class=1,
        train_per_class=2,
    )

    eval_rows = splits[splits["split"] == "eval_strict"]
    assert eval_rows["label"].value_counts().to_dict() == {"21": 1, "22": 1, "31": 1}
    assert set(eval_rows["tier"]) == {"eval_strict"}

    train_rows = splits[splits["split"] == "train"]
    assert {"all", "spatial_only", "quality_spatial", "text_spatial_agreement"}.issubset(
        set(train_rows["tier"])
    )
    assert not set(eval_rows["pageid"]).intersection(set(train_rows["pageid"]))
    agreement_rows = train_rows[train_rows["tier"] == "text_spatial_agreement"]
    assert agreement_rows["qwen_prediction"].eq(agreement_rows["label"]).all()
    assert agreement_rows["gemma_prediction"].eq(agreement_rows["label"]).all()


def test_build_clip_label_splits_rejects_non_positive_per_class_counts(tmp_path: Path) -> None:
    quality_path = tmp_path / "quality.csv"
    qwen_path = tmp_path / "qwen.json"
    gemma_path = tmp_path / "gemma.json"
    pd.DataFrame(
        [
            {
                "pageid": "1",
                "corine_label": "21",
                "quality_bin": "quality_high",
                "landcover_relevance": "high",
                "uncertainty": "low",
                "point_label_share_250m": 0.9,
                "dominant_matches_point_label_250m": True,
            }
        ]
    ).to_csv(quality_path, index=False)
    _write_predictions(qwen_path, {"1": "21"})
    _write_predictions(gemma_path, {"1": "21"})

    with pytest.raises(ValueError, match="per-class counts must be positive"):
        build_clip_label_splits(
            quality_scores_path=quality_path,
            qwen_predictions_path=qwen_path,
            gemma_predictions_path=gemma_path,
            seed=11,
            eval_per_class=0,
            train_per_class=1,
        )
