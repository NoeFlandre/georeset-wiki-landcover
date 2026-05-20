from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from georeset_wiki_landcover.vision.image_probe_splits import (
    add_geo_groups,
    build_image_probe_splits_v2,
    compute_quality_weights,
    tier_mask,
)


def _write_prediction_json(path: Path, predictions: dict[str, str]) -> None:
    path.write_text(
        "{"
        + ",".join(
            f'"{pageid}":{{"prediction":"{prediction}","parse_status":"ok"}}'
            for pageid, prediction in predictions.items()
        )
        + "}",
        encoding="utf-8",
    )


def test_compute_quality_weights_matches_documented_components() -> None:
    frame = pd.DataFrame(
        [
            {
                "pageid": "1",
                "label": "21",
                "landcover_relevance": "high",
                "uncertainty": "low",
                "point_label_share_250m": 0.9,
                "qwen_prediction": "21",
                "gemma_prediction": "21",
            },
            {
                "pageid": "2",
                "label": "21",
                "landcover_relevance": "low",
                "uncertainty": "high",
                "point_label_share_250m": 0.1,
                "qwen_prediction": "22",
                "gemma_prediction": "22",
            },
        ]
    )

    weighted = compute_quality_weights(frame)

    raw_before_norm = np.array([1.25 * 1.0 * 0.9 * 1.25, 0.60 * 0.40 * 0.20 * 0.75])
    expected = raw_before_norm / raw_before_norm.mean()
    assert np.allclose(weighted["weight_raw"], expected)
    assert weighted["weight_class_balanced"].mean() == 1.0


def test_build_image_probe_splits_excludes_eval_from_training_and_keeps_agreement_out_of_eval(
    tmp_path: Path,
) -> None:
    quality_path = tmp_path / "quality.csv"
    qwen_path = tmp_path / "qwen.json"
    gemma_path = tmp_path / "gemma.json"
    rows = []
    qwen: dict[str, str] = {}
    gemma: dict[str, str] = {}
    for label in ["21", "22"]:
        for index in range(6):
            pageid = f"{label}{index}"
            rows.append(
                {
                    "pageid": pageid,
                    "title": pageid,
                    "lat": 48.0 + index / 100,
                    "lon": 2.0 + index / 100,
                    "corine_label": label,
                    "quality_bin": "quality_high",
                    "landcover_relevance": "high",
                    "uncertainty": "low",
                    "point_label_share_250m": 0.95,
                    "dominant_matches_point_label_250m": "true",
                }
            )
            qwen[pageid] = label
            gemma[pageid] = label if index != 0 else "31"
    pd.DataFrame(rows).to_csv(quality_path, index=False)
    _write_prediction_json(qwen_path, qwen)
    _write_prediction_json(gemma_path, gemma)

    splits, weights, manifest = build_image_probe_splits_v2(
        quality_scores_path=quality_path,
        qwen_predictions_path=qwen_path,
        gemma_predictions_path=gemma_path,
        seed=7,
        eval_per_class=1,
        n_repeated_splits=2,
    )

    eval_rows = splits[splits["split"] == "eval_strict"]
    train_rows = splits[splits["split"] == "train"]
    eval_splits = [
        str(split)
        for split in splits["split"].unique()
        if str(split) == "eval_strict"
        or str(split).startswith("repeated_eval_seed_")
        or str(split).startswith("spatial_block_fold_")
    ]
    assert not set(eval_rows["pageid"]).intersection(set(train_rows["pageid"]))
    for split in eval_splits:
        split_eval_pageids = set(splits.loc[splits["split"].eq(split), "pageid"])
        split_train_rows = splits[splits["split"].eq(f"train_for_{split}")]
        if split_eval_pageids == set(splits["pageid"]):
            assert split_train_rows.empty
            continue
        assert not split_train_rows.empty
        assert not split_eval_pageids.intersection(set(split_train_rows["pageid"]))
        assert {"all", "spatial_only", "quality_spatial", "text_spatial_agreement"}.issubset(
            set(split_train_rows["tier"])
        )
    assert set(eval_rows["tier"]) == {"eval_strict"}
    assert "models_agree_with_label" in eval_rows.columns
    assert {"all", "spatial_only", "quality_spatial", "text_spatial_agreement"}.issubset(
        set(train_rows["tier"])
    )
    assert np.isclose(weights["weight_raw"].mean(), 1.0)
    assert manifest["evaluation_selection_excludes_model_agreement"] is True


def test_tier_mask_does_not_treat_false_strings_as_true() -> None:
    frame = pd.DataFrame(
        {
            "spatial_only": ["true", "false", "", None, True, False],
            "quality_spatial": ["yes", "no", "maybe", "1", "0", False],
            "text_spatial_agreement": [1, 0, "TRUE", "FALSE", "nan", None],
        }
    )

    assert tier_mask(frame, "spatial_only").tolist() == [True, False, False, False, True, False]
    assert tier_mask(frame, "quality_spatial").tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert tier_mask(frame, "text_spatial_agreement").tolist() == [
        True,
        False,
        True,
        False,
        False,
        False,
    ]


def test_add_geo_groups_fails_instead_of_silently_collapsing_groups() -> None:
    frame = pd.DataFrame(
        [
            {"pageid": "1", "lat": "not-a-lat", "lon": 2.0},
            {"pageid": "2", "lat": 48.0, "lon": 2.1},
        ]
    )

    with pytest.raises(ValueError, match="Cannot compute EPSG:2154 geo groups"):
        add_geo_groups(frame)
