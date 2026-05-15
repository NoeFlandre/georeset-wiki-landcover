"""Build deterministic split tiers for the CLIP weak-label experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from georeset.vision.clip_weak_labels import write_clip_label_splits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-scores-path",
        type=Path,
        default=Path("data/experiments/article_text_supervision_quality_score_v1/quality_scores.csv"),
    )
    parser.add_argument(
        "--qwen-predictions-path",
        type=Path,
        default=Path(
            "data/experiments/article_text_classification_e2e_with_shuffled_control_v1/"
            "corine_level2_content_predictions.json"
        ),
    )
    parser.add_argument(
        "--gemma-predictions-path",
        type=Path,
        default=Path(
            "data/experiments/article_text_classification_e2e_with_shuffled_control_v1__"
            "gemma4_31b_it_q4_0/corine_level2_content_predictions.json"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/experiments/clip_linear_probe_weak_labels_v1/label_splits.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-class", type=int, default=5)
    parser.add_argument("--train-per-class", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    write_clip_label_splits(
        output_path=args.output_path,
        quality_scores_path=args.quality_scores_path,
        qwen_predictions_path=args.qwen_predictions_path,
        gemma_predictions_path=args.gemma_predictions_path,
        seed=args.seed,
        eval_per_class=args.eval_per_class,
        train_per_class=args.train_per_class,
    )


if __name__ == "__main__":
    main()
