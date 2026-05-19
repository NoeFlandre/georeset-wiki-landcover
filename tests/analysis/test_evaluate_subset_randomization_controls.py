import json

import pandas as pd

from georeset.cli.analysis.evaluate_subset_randomization_controls import (
    SHUFFLED_PAIRS,
    SUBSET_REGISTRY,
    build_random_control_summary,
    compute_metric_row,
    compute_shuffled_delta_row,
    stable_seed,
    target_key,
    target_matched_sample_pageids,
    uniform_sample_pageids,
    write_outputs,
)


def _toy_metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pageid": "1",
                "landcover_relevance": "high",
                "evidence_sentences_count": 2,
                "uncertainty": "low",
                "point_label_share_250m": 0.92,
                "point_label_share_500m": 0.85,
                "dominant_matches_point_label_250m": True,
                "dominant_matches_point_label_500m": True,
                "quality_bin": "quality_very_high",
                "recommended_use": "use_for_training",
                "primary_article_type": "natural_landscape",
            },
            {
                "pageid": "2",
                "landcover_relevance": "medium",
                "evidence_sentences_count": 1,
                "uncertainty": "medium",
                "point_label_share_250m": 0.81,
                "point_label_share_500m": 0.40,
                "dominant_matches_point_label_250m": False,
                "dominant_matches_point_label_500m": False,
                "quality_bin": "quality_high",
                "recommended_use": "use_for_evaluation_only",
                "primary_article_type": "other_or_unclear",
            },
            {
                "pageid": "3",
                "landcover_relevance": "low",
                "evidence_sentences_count": 0,
                "uncertainty": "high",
                "point_label_share_250m": 0.30,
                "point_label_share_500m": 0.20,
                "dominant_matches_point_label_250m": False,
                "dominant_matches_point_label_500m": False,
                "quality_bin": "quality_medium",
                "recommended_use": "exclude",
                "primary_article_type": "settlement_or_administrative",
            },
            {
                "pageid": "4",
                "landcover_relevance": None,
                "evidence_sentences_count": None,
                "uncertainty": None,
                "point_label_share_250m": None,
                "point_label_share_500m": None,
                "dominant_matches_point_label_250m": None,
                "dominant_matches_point_label_500m": None,
                "quality_bin": None,
                "recommended_use": None,
                "primary_article_type": None,
            },
        ]
    )


def _toy_corine_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pageid": "1",
                "target": "31",
                "prediction": "31",
                "parse_status": "ok",
                "task": "corine_level2",
                "text_source": "content",
            },
            {
                "pageid": "2",
                "target": "21",
                "prediction": "31",
                "parse_status": "ok",
                "task": "corine_level2",
                "text_source": "content",
            },
            {
                "pageid": "3",
                "target": "21",
                "prediction": "21",
                "parse_status": "ok",
                "task": "corine_level2",
                "text_source": "content",
            },
        ]
    )


def _toy_osm_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pageid": "1",
                "target": ["wood"],
                "prediction": ["wood"],
                "parse_status": "ok",
                "task": "osm",
                "text_source": "content",
            },
            {
                "pageid": "2",
                "target": ["water", "wood"],
                "prediction": ["water"],
                "parse_status": "ok",
                "task": "osm",
                "text_source": "content",
            },
            {
                "pageid": "3",
                "target": ["water", "wood"],
                "prediction": [],
                "parse_status": "ok",
                "task": "osm",
                "text_source": "content",
            },
        ]
    )


def test_subset_registry_required_masks_and_missing_metadata_are_false() -> None:
    frame = _toy_metadata_frame()

    assert frame.loc[SUBSET_REGISTRY["all"].mask(frame), "pageid"].tolist() == ["1", "2", "3", "4"]
    assert frame.loc[SUBSET_REGISTRY["relevance_medium_high"].mask(frame), "pageid"].tolist() == [
        "1",
        "2",
    ]
    assert frame.loc[SUBSET_REGISTRY["point_label_share_250m_ge_0.8"].mask(frame), "pageid"].tolist() == [
        "1",
        "2",
    ]
    assert frame.loc[
        SUBSET_REGISTRY["relevance_medium_high_and_spatial_250m_ge_0.8"].mask(frame),
        "pageid",
    ].tolist() == ["1", "2"]
    assert frame.loc[
        SUBSET_REGISTRY["quality_high_or_very_high_and_spatial_250m_ge_0.8"].mask(frame),
        "pageid",
    ].tolist() == ["1", "2"]
    assert frame.loc[SUBSET_REGISTRY["recommended_use_training"].mask(frame), "pageid"].tolist() == [
        "1"
    ]
    assert frame.loc[
        SUBSET_REGISTRY["article_type_high_prior_and_relevance_medium_high"].mask(frame),
        "pageid",
    ].tolist() == ["1"]


def test_uniform_sampler_exact_n_no_replacement_and_stable_independent_seed() -> None:
    universe = ["1", "2", "3", "4", "5"]
    seed_a = stable_seed(42, "parent", "qwen", "corine_level2", "content", "subset_a", "random_same_n")
    seed_b = stable_seed(42, "parent", "qwen", "corine_level2", "content", "subset_b", "random_same_n")

    draw_a = uniform_sample_pageids(universe, 3, seed_a)
    assert draw_a == uniform_sample_pageids(list(reversed(universe)), 3, seed_a)
    assert len(draw_a) == 3
    assert len(set(draw_a)) == 3
    assert draw_a != uniform_sample_pageids(universe, 3, seed_b)


def test_target_matched_sampler_matches_corine_and_osm_or_skips() -> None:
    corine_universe = _toy_corine_records()
    corine_observed = corine_universe[corine_universe["pageid"].isin(["1", "2"])]
    result = target_matched_sample_pageids(corine_universe, corine_observed, seed=123)

    assert result.status == "ok"
    sampled = corine_universe[corine_universe["pageid"].isin(result.pageids)]
    assert sampled["target"].value_counts().to_dict() == corine_observed["target"].value_counts().to_dict()

    osm_universe = _toy_osm_records()
    osm_observed = osm_universe[osm_universe["pageid"].isin(["1", "2"])]
    result = target_matched_sample_pageids(osm_universe, osm_observed, seed=123)
    assert result.status == "ok"
    assert [target_key(value) for value in osm_observed["target"]] == [
        target_key(value) for value in osm_universe[osm_universe["pageid"].isin(result.pageids)]["target"]
    ]

    impossible = target_matched_sample_pageids(corine_universe.iloc[:1], corine_observed, seed=123)
    assert impossible.status == "skipped_insufficient_target_support"
    assert impossible.pageids == []


def test_metric_recomputation_uses_selected_pageids_and_task_primary_metrics() -> None:
    row = compute_metric_row(
        records=_toy_corine_records(),
        selected_pageids=["1", "2"],
        labels=["21", "31"],
        task="corine_level2",
        primary_metric="balanced_accuracy",
    )
    assert row["n"] == 2
    assert row["observed_primary_score"] == 0.5
    assert row["majority_balanced_accuracy"] == 0.5
    assert row["unstable_small_n"] is True

    osm = compute_metric_row(
        records=_toy_osm_records(),
        selected_pageids=["1", "2"],
        labels=["water", "wood"],
        task="osm",
        primary_metric="jaccard",
    )
    assert osm["n"] == 2
    assert osm["primary_metric"] == "jaccard"
    assert "micro_f1" in osm
    assert "exact_match_accuracy" in osm
    assert "hamming_loss" in osm


def test_random_control_summary_contains_percentiles_and_small_n_flag() -> None:
    observed = compute_metric_row(
        records=_toy_corine_records(),
        selected_pageids=["1", "2"],
        labels=["21", "31"],
        task="corine_level2",
        primary_metric="balanced_accuracy",
    )
    summary = build_random_control_summary(
        observed_row=observed,
        random_scores=[0.1, 0.2, 0.5, 0.7],
        n_draws_requested=4,
        control_type="random_same_n",
    )
    assert summary["random_mean"] == 0.375
    assert summary["observed_percentile"] == 75.0
    assert summary["empirical_p_greater_equal"] == 0.6
    assert summary["unstable_small_n"] is True


def test_shuffled_delta_uses_same_sampled_pageids() -> None:
    aligned = _toy_corine_records()
    shuffled = aligned.copy()
    shuffled["text_source"] = "content_shuffled"
    shuffled["prediction"] = ["21", "21", "21"]

    row = compute_shuffled_delta_row(
        aligned_records=aligned,
        shuffled_records=shuffled,
        pageids=["1", "2"],
        labels=["21", "31"],
        task="corine_level2",
        primary_metric="balanced_accuracy",
    )
    assert row["observed_aligned_score"] == 0.5
    assert row["observed_shuffled_score"] == 0.5
    assert row["observed_delta"] == 0.0
    assert row["sampled_pageids"] == "1,2"
    assert SHUFFLED_PAIRS["content"] == "content_shuffled"


def test_outputs_and_manifest_are_written_without_parent_mutation(tmp_path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    marker = parent / "README.md"
    marker.write_text("frozen", encoding="utf-8")

    output_dir = tmp_path / "out"
    write_outputs(
        output_dir=output_dir,
        observed_rows=[{"parent_experiment_id": "parent", "n": 2}],
        random_same_n_rows=[{"control_type": "random_same_n", "n": 2}],
        random_target_rows=[{"control_type": "random_same_target_distribution", "n": 2}],
        shuffled_delta_rows=[{"text_source": "content", "observed_delta": 0.1}],
        class_distribution_rows=[{"target_key": "31", "support": 2}],
        significant_rows=[{"subset_name": "relevance_medium_high", "conclusion_flag": "beats_same_n"}],
        manifest={
            "experiment_id": "article_text_subset_randomization_controls_v1",
            "no_llm_rerun": True,
            "comparison_universe": "same parent/model/task/text_source plus subset metadata availability",
            "parent_experiments_used": [str(parent)],
        },
        summary_text="# Summary\n\nNo LLM rerun.\n",
    )

    for filename in [
        "observed_subset_metrics.csv",
        "observed_subset_metrics.md",
        "random_same_n_controls.csv",
        "random_same_n_controls.md",
        "random_target_matched_controls.csv",
        "random_target_matched_controls.md",
        "shuffled_delta_random_controls.csv",
        "shuffled_delta_random_controls.md",
        "subset_class_distribution.csv",
        "subset_class_distribution.md",
        "significant_filter_summary.csv",
        "significant_filter_summary.md",
        "manifest.json",
        "summary.md",
    ]:
        assert (output_dir / filename).exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_llm_rerun"] is True
    assert marker.read_text(encoding="utf-8") == "frozen"
