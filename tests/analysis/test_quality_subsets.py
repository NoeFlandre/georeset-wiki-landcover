import pandas as pd

from georeset.analysis.quality_subsets import quality_subset_masks


def test_quality_subset_masks_select_expected_rows() -> None:
    records = pd.DataFrame(
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
                "landcover_relevance": "medium",
                "point_label_share_250m": 0.7,
                "quality_bin": "quality_very_high",
                "recommended_use": "use_for_evaluation_only",
            },
            {
                "pageid": "3",
                "landcover_relevance": "none",
                "point_label_share_250m": 0.1,
                "quality_bin": "quality_low",
                "recommended_use": "exclude",
            },
        ]
    )

    masks = quality_subset_masks(records)

    assert records.loc[masks["all"], "pageid"].tolist() == ["1", "2", "3"]
    assert records.loc[masks["relevance_medium_high"], "pageid"].tolist() == ["1", "2"]
    assert records.loc[masks["spatial_250m_ge_0.8"], "pageid"].tolist() == ["1"]
    assert records.loc[
        masks["relevance_medium_high_and_spatial_250m_ge_0.8"], "pageid"
    ].tolist() == ["1"]
    assert records.loc[masks["quality_high_or_very_high"], "pageid"].tolist() == ["1", "2"]
    assert records.loc[
        masks["quality_high_or_very_high_and_spatial_250m_ge_0.8"], "pageid"
    ].tolist() == ["1"]
    assert records.loc[masks["recommended_use_training"], "pageid"].tolist() == ["1"]
    assert records.loc[masks["recommended_use_evaluation_only"], "pageid"].tolist() == ["2"]
    assert records.loc[masks["recommended_use_exclude"], "pageid"].tolist() == ["3"]


def test_quality_subset_masks_handle_missing_metadata_columns() -> None:
    records = pd.DataFrame([{"pageid": "1"}, {"pageid": "2"}])

    masks = quality_subset_masks(records)

    assert records.loc[masks["all"], "pageid"].tolist() == ["1", "2"]
    assert not masks["quality_high_or_very_high"].any()
    assert not masks["recommended_use_training"].any()
