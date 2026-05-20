"""Classification runner orchestration."""

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import Any, Protocol, cast

from georeset_wiki_landcover.classification.llm_classifier import LLMClassifier
from georeset_wiki_landcover.classification.metrics import multilabel_metrics, single_label_metrics
from georeset_wiki_landcover.classification.records import (
    build_prediction_record,
    should_skip_record,
)
from georeset_wiki_landcover.classification.task_setup import load_task_setup
from georeset_wiki_landcover.classification.text_sources import (
    SHUFFLED_TEXT_SOURCES,
    TEXT_SOURCE_CHOICES,
    apply_shuffled_text_control,
    base_text_source,
    shuffled_metadata,
)
from georeset_wiki_landcover.classification.types import PredictionRecord, PredictionResult
from georeset_wiki_landcover.config import DataPaths, ModelSettings
from georeset_wiki_landcover.contracts import (
    ArticleMeta,
    ClassificationTarget,
    MultiLabelMetricResult,
    SingleLabelMetricResult,
)
from georeset_wiki_landcover.utils.articles import index_articles_by_pageid
from georeset_wiki_landcover.utils.json_io import read_json_file, write_json_atomic

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

CLASSIFICATION_POLICY_VERSION = 4


class Classifier(Protocol):
    def classify_single_label(
        self,
        text: str,
        allowed_labels: list[str],
        label_descriptions: dict[str, str],
        task: str,
        text_source: str,
    ) -> PredictionResult: ...

    def classify_multilabel(
        self,
        text: str,
        allowed_labels: list[str],
        task: str,
        text_source: str,
    ) -> PredictionResult: ...


ClassifierFactory = Callable[[str | None, str | None, int, float], Classifier]


def default_classifier_factory(
    model_path: str | None, model_repo_id: str | None, seed: int, temperature: float
) -> Classifier:
    return cast(
        Classifier,
        LLMClassifier(
            model_path=model_path,
            model_repo_id=model_repo_id,
            seed=seed,
            temperature=temperature,
        ),
    )


def prediction_fingerprint(
    task: str,
    text_source: str,
    model_path: str,
    model_repo_id: str | None,
    seed: int,
    temperature: float,
    allowed_labels: list[str],
) -> str:
    payload = {
        "task": task,
        "text_source": text_source,
        "model": model_path,
        "model_repo_id": model_repo_id,
        "seed": seed,
        "temperature": temperature,
        "allowed_labels": sorted(allowed_labels),
        "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def text_fingerprint(text: str) -> str:
    """Hash the exact text sent to the classifier for checkpoint invalidation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_paths = DataPaths()
    model_settings = ModelSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Classify Wikipedia articles into land-cover labels."
    )
    parser.add_argument("--task", choices=["corine_level2", "osm"], default="corine_level2")
    parser.add_argument(
        "--text-source",
        choices=TEXT_SOURCE_CHOICES,
        default="summary",
    )
    parser.add_argument("--wiki-articles-path", default=data_paths.wiki_articles)
    parser.add_argument("--article-contents-path", default=data_paths.article_contents)
    parser.add_argument("--article-summaries-path", default=data_paths.article_summaries)
    parser.add_argument(
        "--article-summaries-no-place-path",
        default=data_paths.article_summaries_no_place,
    )
    parser.add_argument(
        "--article-landuse-evidence-summaries-path",
        default=data_paths.article_landuse_evidence_summaries,
    )
    parser.add_argument("--article-evidence-cards-path", default=data_paths.article_evidence_cards)
    parser.add_argument(
        "--article-evidence-highlights-path", default=data_paths.article_evidence_highlights
    )
    parser.add_argument(
        "--article-retrieved-evidence-windows-path",
        default=data_paths.article_retrieved_evidence_windows,
    )
    parser.add_argument("--osm-polygons-path", default=data_paths.osm_polygons)
    parser.add_argument(
        "--corine-polygons-path",
        default=data_paths.corine_polygons,
    )
    parser.add_argument("--output-dir", default=data_paths.classification_output_dir)
    parser.add_argument(
        "--model-path",
        default=model_settings.model_path,
    )
    parser.add_argument(
        "--model-repo-id",
        default=model_settings.model_repo_id,
        help="Optional Hugging Face repo_id for llama_cpp.Llama.from_pretrained.",
    )
    parser.add_argument("--seed", type=int, default=model_settings.seed)
    parser.add_argument(
        "--temperature", type=float, default=model_settings.classification_temperature
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def load_text_source(
    text_source: str,
    article_contents_path: str,
    article_summaries_path: str,
    article_summaries_no_place_path: str,
    article_landuse_evidence_summaries_path: str | None = None,
    article_evidence_cards_path: str | None = None,
    article_evidence_highlights_path: str | None = None,
    article_retrieved_evidence_windows_path: str | None = None,
) -> dict[str, str]:
    text_source = base_text_source(text_source)
    if text_source == "summary":
        data = read_json_file(article_summaries_path)
        return {k: v["summary"] for k, v in data.items()}
    elif text_source == "summary_no_place":
        data = read_json_file(article_summaries_no_place_path)
        return {k: v["summary"] for k, v in data.items()}
    elif text_source == "content":
        data = read_json_file(article_contents_path)
        return {k: v.get("content", "") for k, v in data.items()}
    elif text_source == "landuse_evidence_summary":
        if article_landuse_evidence_summaries_path is None:
            raise ValueError("landuse evidence summaries path is required")
        data = read_json_file(article_landuse_evidence_summaries_path)
        return {k: v["landuse_evidence_summary"] for k, v in data.items()}
    elif text_source == "evidence_card":
        if article_evidence_cards_path is None:
            raise ValueError("evidence cards path is required")
        data = read_json_file(article_evidence_cards_path)
        return {k: v["evidence_card"] for k, v in data.items()}
    elif text_source == "content_with_evidence_card":
        if article_evidence_cards_path is None:
            raise ValueError("evidence cards path is required")
        data = read_json_file(article_evidence_cards_path)
        return {k: v["content_with_evidence_card"] for k, v in data.items()}
    elif text_source == "content_with_evidence_highlights":
        if article_evidence_highlights_path is None:
            raise ValueError("evidence highlights path is required")
        data = read_json_file(article_evidence_highlights_path)
        return {k: v["content_with_evidence_highlights"] for k, v in data.items()}
    elif text_source in {
        "retrieved_evidence_windows",
        "retrieved_evidence_sentences_only",
        "random_sentence_windows",
        "retrieved_evidence_windows_no_place",
    }:
        if article_retrieved_evidence_windows_path is None:
            raise ValueError("retrieved evidence windows path is required")
        data = read_json_file(article_retrieved_evidence_windows_path)
        return {k: v[text_source] for k, v in data.items()}
    else:
        raise ValueError(f"Unknown text source: {text_source}")


def compute_metrics(
    task: str,
    text_source: str,
    y_true: dict[str, ClassificationTarget],
    y_pred: dict[str, ClassificationTarget],
    allowed_labels: list[str],
) -> tuple[SingleLabelMetricResult | MultiLabelMetricResult, list[str]]:
    """Returns (metrics_dict, labels_evaluated)."""
    eval_labels = sorted(
        {v for vals in y_true.values() for v in (vals if isinstance(vals, list) else [vals])}
        | {v for vals in y_pred.values() for v in (vals if isinstance(vals, list) else [vals])}
    )
    metrics: SingleLabelMetricResult | MultiLabelMetricResult
    if task == "corine_level2":
        metrics = single_label_metrics(
            {key: str(value) for key, value in y_true.items()},
            {key: str(value) for key, value in y_pred.items()},
            labels=eval_labels,
        )
    else:
        metrics = multilabel_metrics(
            {
                key: [str(item) for item in value] if isinstance(value, list) else [str(value)]
                for key, value in y_true.items()
            },
            {
                key: [str(item) for item in value] if isinstance(value, list) else [str(value)]
                for key, value in y_pred.items()
            },
            labels=eval_labels,
        )
    metrics["task"] = task
    metrics["text_source"] = text_source
    metrics["allowed_labels"] = sorted(allowed_labels)
    metrics["labels_evaluated"] = eval_labels
    return metrics, eval_labels


def main(
    argv: list[str] | None = None,
    *,
    classifier_factory: ClassifierFactory = default_classifier_factory,
) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    out_pred = os.path.join(args.output_dir, f"{args.task}_{args.text_source}_predictions.json")
    out_metrics = os.path.join(args.output_dir, f"{args.task}_{args.text_source}_metrics.json")

    articles = cast(list[ArticleMeta], read_json_file(args.wiki_articles_path))
    articles_by_pageid = index_articles_by_pageid(articles)

    text_records = load_text_source(
        args.text_source,
        args.article_contents_path,
        args.article_summaries_path,
        args.article_summaries_no_place_path,
        args.article_landuse_evidence_summaries_path,
        args.article_evidence_cards_path,
        args.article_evidence_highlights_path,
        args.article_retrieved_evidence_windows_path,
    )

    task_setup = load_task_setup(
        task=args.task,
        articles=articles,
        corine_polygons_path=args.corine_polygons_path,
        osm_polygons_path=args.osm_polygons_path,
    )
    target = task_setup.target
    allowed_labels = task_setup.allowed_labels
    label_descriptions = task_setup.label_descriptions

    eligible = [pid for pid in text_records if pid in target]
    if args.limit:
        eligible = eligible[: args.limit]
    shuffled_source_pageids: dict[str, str] = {}
    if args.text_source in SHUFFLED_TEXT_SOURCES:
        text_records, shuffled_source_pageids = apply_shuffled_text_control(
            text_records, eligible, seed=args.seed
        )

    existing: dict[str, PredictionRecord] = {}
    if os.path.exists(out_pred):
        existing = cast(dict[str, PredictionRecord], read_json_file(out_pred))

    fp_current = prediction_fingerprint(
        args.task,
        args.text_source,
        args.model_path,
        args.model_repo_id,
        args.seed,
        args.temperature,
        allowed_labels,
    )

    classifier = classifier_factory(
        args.model_path, args.model_repo_id, args.seed, args.temperature
    )

    for pageid in eligible:
        previous_record = cast(dict[str, Any] | None, existing.get(pageid))
        text = text_records[pageid]
        text_sha256 = text_fingerprint(text)
        if should_skip_record(
            previous_record,
            fp_current,
            text_sha256=text_sha256,
            retry_failed=args.retry_failed,
        ):
            continue
        article = articles_by_pageid.get(pageid)
        title = article.get("title", pageid) if article else pageid
        if args.task == "corine_level2":
            result = classifier.classify_single_label(
                text=text,
                allowed_labels=allowed_labels,
                label_descriptions=label_descriptions,
                task=args.task,
                text_source=args.text_source,
            )
        else:
            result = classifier.classify_multilabel(
                text=text,
                allowed_labels=allowed_labels,
                task=args.task,
                text_source=args.text_source,
            )
        extra_metadata = None
        if shuffled_source_pageids:
            extra_metadata = shuffled_metadata(args.text_source, shuffled_source_pageids[pageid])
        prediction_record = build_prediction_record(
            pageid=pageid,
            title=title,
            target=target[pageid],
            result=result,
            fingerprint=fp_current,
            text_sha256=text_sha256,
            extra_metadata=extra_metadata,
        )
        existing[pageid] = prediction_record
        write_json_atomic(out_pred, existing, indent=2, ensure_ascii=False)
        status = result.get("parse_status", "error")
        logger.info(f"[{len(existing)}/{len(eligible)}] {pageid} ({title}): {status}")

    y_pred: dict[str, ClassificationTarget] = {
        k: cast(ClassificationTarget, v["prediction"])
        for k, v in existing.items()
        if k in eligible and v.get("parse_status") == "ok"
    }
    y_true = {k: target[k] for k in eligible}
    metrics, _ = compute_metrics(args.task, args.text_source, y_true, y_pred, allowed_labels)
    write_json_atomic(out_metrics, metrics, indent=2, ensure_ascii=False)
    logger.info(f"Wrote metrics to {out_metrics}")


if __name__ == "__main__":
    main()
