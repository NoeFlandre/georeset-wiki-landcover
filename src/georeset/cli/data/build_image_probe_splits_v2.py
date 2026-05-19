"""Build Experiment 014 image probe splits and sample weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from georeset.cli.image_probe_args import (
    image_probe_splits_path,
    sample_weights_path,
    split_manifest_path,
    split_summary_path,
)
from georeset.experiment_paths import experiment_artifact_dir, experiment_artifact_file
from georeset.utils.json_io import (
    markdown_table,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from georeset.vision.image_probe_splits import build_image_probe_splits_v2

EXPERIMENT_ID = "quality_weighted_multiscale_image_probe_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-scores-path",
        type=Path,
        default=experiment_artifact_file(
            "article_text_supervision_quality_score_v1", "quality_scores.csv"
        ),
    )
    parser.add_argument(
        "--qwen-predictions-path",
        type=Path,
        default=experiment_artifact_file(
            "article_text_classification_e2e_with_shuffled_control_v1",
            "corine_level2_content_predictions.json",
        ),
    )
    parser.add_argument(
        "--gemma-predictions-path",
        type=Path,
        default=experiment_artifact_file(
            "article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0",
            "corine_level2_content_predictions.json",
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=experiment_artifact_dir(EXPERIMENT_ID))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-class", type=int, default=5)
    parser.add_argument("--n-repeated-splits", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    splits, weights, manifest = build_image_probe_splits_v2(
        quality_scores_path=args.quality_scores_path,
        qwen_predictions_path=args.qwen_predictions_path,
        gemma_predictions_path=args.gemma_predictions_path,
        seed=args.seed,
        eval_per_class=args.eval_per_class,
        n_repeated_splits=args.n_repeated_splits,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(image_probe_splits_path(args.output_dir), splits, index=False)
    write_csv_atomic(sample_weights_path(args.output_dir), weights, index=False)
    write_json_atomic(
        split_manifest_path(args.output_dir),
        {
            **manifest,
            "quality_scores_path": str(args.quality_scores_path),
            "qwen_predictions_path": str(args.qwen_predictions_path),
            "gemma_predictions_path": str(args.gemma_predictions_path),
        },
    )
    tier_counts = splits.groupby(["split", "tier"]).size().reset_index(name="n")
    write_text_atomic(
        split_summary_path(args.output_dir),
        "# quality_weighted_multiscale_image_probe_v1 splits\n\n"
        "Evaluation rows are selected from `quality_spatial`; Qwen/Gemma agreement is used only "
        "for training weights and the `text_spatial_agreement` training tier.\n\n"
        + markdown_table(rows=tier_counts.to_dict("records"))
        + "\n",
    )


if __name__ == "__main__":
    main()
