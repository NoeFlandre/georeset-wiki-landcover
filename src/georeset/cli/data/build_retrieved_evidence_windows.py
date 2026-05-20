"""Build deterministic retrieved-evidence sentence-window text artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from georeset.cli.data.json_inputs import (
    index_json_records_by_pageid,
    read_optional_json_mapping,
    read_required_json_mapping,
)
from georeset.config import DataPaths, ModelSettings
from georeset.text.retrieved_evidence_windows import (
    RETRIEVED_EVIDENCE_WINDOWS_VERSION,
    build_retrieved_evidence_window_record,
)
from georeset.utils.json_io import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_paths = DataPaths()
    model_settings = ModelSettings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-contents-path", type=Path, default=data_paths.article_contents)
    parser.add_argument(
        "--evidence-metadata-path",
        type=Path,
        default=data_paths.article_landuse_evidence_summaries,
    )
    parser.add_argument(
        "--output-path", type=Path, default=data_paths.article_retrieved_evidence_windows
    )
    parser.add_argument("--seed", type=int, default=model_settings.seed)
    return parser.parse_args(argv)


def build_retrieved_evidence_windows(
    *,
    article_contents_path: Path,
    evidence_metadata_path: Path,
    seed: int,
) -> dict[str, dict[str, Any]]:
    articles = read_required_json_mapping(article_contents_path, description="article contents")
    evidence_by_pageid = index_json_records_by_pageid(
        read_optional_json_mapping(evidence_metadata_path)
    )
    output: dict[str, dict[str, Any]] = {}
    for pageid, article in sorted(articles.items(), key=lambda item: str(item[0])):
        if not isinstance(article, dict):
            continue
        pageid_str = str(pageid)
        output[pageid_str] = build_retrieved_evidence_window_record(
            pageid=pageid_str,
            article=article,
            evidence=evidence_by_pageid.get(pageid_str),
            seed=seed,
        )
    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = build_retrieved_evidence_windows(
        article_contents_path=args.article_contents_path,
        evidence_metadata_path=args.evidence_metadata_path,
        seed=args.seed,
    )
    payload = {
        pageid: {
            **record,
            "metadata": {
                **record["metadata"],
                "version": RETRIEVED_EVIDENCE_WINDOWS_VERSION,
            },
        }
        for pageid, record in records.items()
    }
    write_json_atomic(args.output_path, payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
