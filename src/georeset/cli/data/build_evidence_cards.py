"""Build deterministic evidence-card text artifacts from existing metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from georeset.analysis.article_type_metadata_loading import load_article_type_metadata
from georeset.analysis.pageid_frames import dataframe_by_pageid, load_optional_pageid_csv
from georeset.analysis.spatial_confidence_loading import load_spatial_confidence
from georeset.cli.data.json_inputs import (
    index_json_records_by_pageid,
    read_optional_json_mapping,
    read_required_json_mapping,
)
from georeset.config import DataPaths
from georeset.experiment_paths import experiment_artifact_file
from georeset.text.evidence_cards import EVIDENCE_CARD_VERSION, build_evidence_card_record
from georeset.utils.json_io import write_json_atomic

DEFAULT_ARTICLE_TYPE_METADATA_PATH = experiment_artifact_file(
    "article_text_classification_article_type_relevance_stratified_v1",
    "article_type_assignments.csv",
)
DEFAULT_SPATIAL_CONFIDENCE_PATH = experiment_artifact_file(
    "corine_spatial_confidence_v1", "spatial_confidence.csv"
)
DEFAULT_QUALITY_SCORES_PATH = experiment_artifact_file(
    "article_text_supervision_quality_score_v1", "quality_scores.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_paths = DataPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-contents-path", type=Path, default=data_paths.article_contents)
    parser.add_argument(
        "--evidence-metadata-path",
        type=Path,
        default=data_paths.article_landuse_evidence_summaries,
    )
    parser.add_argument(
        "--article-type-metadata-path",
        type=Path,
        default=DEFAULT_ARTICLE_TYPE_METADATA_PATH,
    )
    parser.add_argument(
        "--spatial-confidence-path",
        type=Path,
        default=DEFAULT_SPATIAL_CONFIDENCE_PATH,
    )
    parser.add_argument("--quality-scores-path", type=Path, default=DEFAULT_QUALITY_SCORES_PATH)
    parser.add_argument("--output-path", type=Path, default=data_paths.article_evidence_cards)
    return parser.parse_args(argv)


def _load_spatial(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_spatial_confidence(path, allow_missing_pageid=True)


def build_cards(
    *,
    article_contents_path: Path,
    evidence_metadata_path: Path,
    article_type_metadata_path: Path,
    spatial_confidence_path: Path,
    quality_scores_path: Path,
) -> dict[str, dict[str, Any]]:
    articles = read_required_json_mapping(
        article_contents_path, description="article contents"
    )
    evidence_by_pageid = index_json_records_by_pageid(
        read_optional_json_mapping(evidence_metadata_path)
    )
    article_types = dataframe_by_pageid(load_article_type_metadata(article_type_metadata_path))
    spatial = dataframe_by_pageid(_load_spatial(spatial_confidence_path))
    quality = dataframe_by_pageid(load_optional_pageid_csv(quality_scores_path))

    output: dict[str, dict[str, Any]] = {}
    for pageid, article in sorted(articles.items(), key=lambda item: str(item[0])):
        if not isinstance(article, dict):
            continue
        pageid_str = str(pageid)
        output[pageid_str] = build_evidence_card_record(
            pageid=pageid_str,
            article=article,
            evidence=evidence_by_pageid.get(pageid_str),
            article_type=article_types.get(pageid_str),
            spatial=spatial.get(pageid_str),
            quality=quality.get(pageid_str),
        )
    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = build_cards(
        article_contents_path=args.article_contents_path,
        evidence_metadata_path=args.evidence_metadata_path,
        article_type_metadata_path=args.article_type_metadata_path,
        spatial_confidence_path=args.spatial_confidence_path,
        quality_scores_path=args.quality_scores_path,
    )
    payload = {
        pageid: {
            **record,
            "metadata": {
                **record["metadata"],
                "version": EVIDENCE_CARD_VERSION,
            },
        }
        for pageid, record in records.items()
    }
    write_json_atomic(args.output_path, payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
