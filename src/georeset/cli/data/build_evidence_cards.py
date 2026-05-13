"""Build deterministic evidence-card text artifacts from existing metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from georeset.analysis.article_type_metadata_loading import load_article_type_metadata
from georeset.analysis.spatial_confidence_loading import load_spatial_confidence
from georeset.config import DataPaths
from georeset.text.evidence_cards import EVIDENCE_CARD_VERSION, build_evidence_card_record
from georeset.utils.json_io import read_json_file, write_json_atomic

DEFAULT_ARTICLE_TYPE_METADATA_PATH = Path(
    "data/experiments/article_text_classification_article_type_relevance_stratified_v1/"
    "article_type_assignments.csv"
)
DEFAULT_SPATIAL_CONFIDENCE_PATH = Path(
    "data/experiments/corine_spatial_confidence_v1/spatial_confidence.csv"
)
DEFAULT_QUALITY_SCORES_PATH = Path(
    "data/experiments/article_text_supervision_quality_score_v1/quality_scores.csv"
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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = read_json_file(path)
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, Any], raw)


def _json_records_by_pageid(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for pageid_key, payload in records.items():
        if not isinstance(payload, dict):
            continue
        pageid = payload.get("pageid", pageid_key)
        if pageid in (None, ""):
            pageid = pageid_key
        indexed[str(pageid)] = payload
    return indexed


def _dataframe_by_pageid(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "pageid" not in frame.columns:
        return {}
    indexed: dict[str, pd.Series] = {}
    for _, row in frame.iterrows():
        indexed[str(row["pageid"])] = row
    return indexed


def _load_spatial(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_spatial_confidence(path, allow_missing_pageid=True)


def _load_quality(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"pageid": str})
    if "pageid" not in frame.columns:
        return pd.DataFrame()
    frame["pageid"] = frame["pageid"].astype(str)
    return frame


def build_cards(
    *,
    article_contents_path: Path,
    evidence_metadata_path: Path,
    article_type_metadata_path: Path,
    spatial_confidence_path: Path,
    quality_scores_path: Path,
) -> dict[str, dict[str, Any]]:
    articles = _read_json_mapping(article_contents_path)
    evidence_by_pageid = _json_records_by_pageid(_read_json_mapping(evidence_metadata_path))
    article_types = _dataframe_by_pageid(load_article_type_metadata(article_type_metadata_path))
    spatial = _dataframe_by_pageid(_load_spatial(spatial_confidence_path))
    quality = _dataframe_by_pageid(_load_quality(quality_scores_path))

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
