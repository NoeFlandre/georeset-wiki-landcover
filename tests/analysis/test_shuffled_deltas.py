from georeset.analysis.shuffled_deltas import compute_shuffled_delta_rows, primary_metric_name


def test_compute_shuffled_delta_rows_matches_with_model_columns_and_scores():
    rows = [
        {
            "subset": "all",
            "model_key": "qwen",
            "model": "Qwen",
            "task": "corine_level2",
            "text_source": "content",
            "balanced_accuracy": 0.4,
            "jaccard": "",
            "n": 10,
        },
        {
            "subset": "all",
            "model_key": "qwen",
            "model": "Qwen",
            "task": "corine_level2",
            "text_source": "content_shuffled",
            "balanced_accuracy": 0.1,
            "jaccard": "",
            "n": 10,
        },
    ]

    deltas = compute_shuffled_delta_rows(
        rows,
        shuffled_pairs={"content": "content_shuffled"},
        model_columns=("model_key", "model"),
    )

    assert deltas == [
        {
            "subset": "all",
            "model_key": "qwen",
            "model": "Qwen",
            "task": "corine_level2",
            "text_source": "content",
            "shuffled_text_source": "content_shuffled",
            "primary_metric": "balanced_accuracy",
            "aligned_score": 0.4,
            "shuffled_score": 0.1,
            "delta": 0.30000000000000004,
            "n_aligned": 10,
            "n_shuffled": 10,
        }
    ]


def test_compute_shuffled_delta_rows_uses_jaccard_for_osm_and_skips_missing_pair():
    rows = [
        {
            "subset": "quality_high",
            "model": "Qwen",
            "task": "osm",
            "text_source": "evidence_card",
            "balanced_accuracy": "",
            "jaccard": 0.25,
            "n": 7,
        },
        {
            "subset": "quality_high",
            "model": "Qwen",
            "task": "osm",
            "text_source": "unrelated",
            "balanced_accuracy": "",
            "jaccard": 0.0,
            "n": 7,
        },
    ]

    assert (
        compute_shuffled_delta_rows(
            rows,
            shuffled_pairs={"evidence_card": "evidence_card_shuffled"},
            model_columns=("model",),
        )
        == []
    )

    rows.append(
        {
            "subset": "quality_high",
            "model": "Qwen",
            "task": "osm",
            "text_source": "evidence_card_shuffled",
            "balanced_accuracy": "",
            "jaccard": 0.1,
            "n": 7,
        }
    )

    [delta] = compute_shuffled_delta_rows(
        rows,
        shuffled_pairs={"evidence_card": "evidence_card_shuffled"},
        model_columns=("model",),
    )

    assert delta["primary_metric"] == "jaccard"
    assert delta["delta"] == 0.15


def test_primary_metric_name_can_use_exact_match_for_osm_reports() -> None:
    assert (
        primary_metric_name("corine_level2", osm_metric="exact_match_accuracy")
        == "balanced_accuracy"
    )
    assert primary_metric_name("osm", osm_metric="exact_match_accuracy") == "exact_match_accuracy"
