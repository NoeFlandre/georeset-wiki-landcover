"""Classification runner orchestration."""

import argparse
import hashlib
import json
import logging
import os

from src.classification.llm_classifier import LLMClassifier
from src.classification.metrics import multilabel_metrics, single_label_metrics
from src.classification.records import build_prediction_record, should_skip_record
from src.classification.task_setup import load_task_setup
from src.classification.text_sources import (
    SHUFFLED_TEXT_SOURCES,
    TEXT_SOURCE_CHOICES,
    apply_shuffled_text_control,
    base_text_source,
    shuffled_metadata,
)
from src.contracts import ArticleMeta, ClassificationTarget, MetricResult
from src.utils.json_io import write_json_atomic

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

CLASSIFICATION_POLICY_VERSION = 4


def prediction_fingerprint(
    task: str,
    text_source: str,
    model_path: str,
    seed: int,
    temperature: float,
    allowed_labels: list[str],
) -> str:
    payload = {
        "task": task,
        "text_source": text_source,
        "model": model_path,
        "seed": seed,
        "temperature": temperature,
        "allowed_labels": sorted(allowed_labels),
        "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Wikipedia articles into land-cover labels."
    )
    parser.add_argument("--task", choices=["corine_level2", "osm"], default="corine_level2")
    parser.add_argument(
        "--text-source",
        choices=TEXT_SOURCE_CHOICES,
        default="summary",
    )
    parser.add_argument("--wiki-articles-path", default="data/wiki/wiki_articles.json")
    parser.add_argument("--article-contents-path", default="data/wiki/article_contents.json")
    parser.add_argument("--article-summaries-path", default="data/wiki/article_summaries.json")
    parser.add_argument(
        "--article-summaries-no-place-path",
        default="data/wiki/article_summaries_no_place.json",
    )
    parser.add_argument("--osm-polygons-path", default="data/osm/osm_project_polygons.geojson")
    parser.add_argument(
        "--corine-polygons-path",
        default="data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp",
    )
    parser.add_argument("--output-dir", default="data/classification")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("GEORESET_MODEL_PATH", "Qwen3.6-27B-Q4_0.gguf"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def load_text_source(
    text_source: str,
    article_contents_path: str,
    article_summaries_path: str,
    article_summaries_no_place_path: str,
) -> dict[str, str]:
    text_source = base_text_source(text_source)
    if text_source == "summary":
        with open(article_summaries_path) as f:
            data = json.load(f)
        return {k: v["summary"] for k, v in data.items()}
    elif text_source == "summary_no_place":
        with open(article_summaries_no_place_path) as f:
            data = json.load(f)
        return {k: v["summary"] for k, v in data.items()}
    elif text_source == "content":
        with open(article_contents_path) as f:
            data = json.load(f)
        return {k: v.get("content", "") for k, v in data.items()}
    else:
        raise ValueError(f"Unknown text source: {text_source}")


def compute_metrics(
    task: str,
    text_source: str,
    y_true: dict[str, ClassificationTarget],
    y_pred: dict[str, ClassificationTarget],
    allowed_labels: list[str],
) -> tuple[MetricResult, list[str]]:
    """Returns (metrics_dict, labels_evaluated)."""
    eval_labels = sorted(
        {v for vals in y_true.values() for v in (vals if isinstance(vals, list) else [vals])}
        | {v for vals in y_pred.values() for v in (vals if isinstance(vals, list) else [vals])}
    )
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    out_pred = os.path.join(args.output_dir, f"{args.task}_{args.text_source}_predictions.json")
    out_metrics = os.path.join(args.output_dir, f"{args.task}_{args.text_source}_metrics.json")

    with open(args.wiki_articles_path) as f:
        articles: list[ArticleMeta] = json.load(f)

    text_records = load_text_source(
        args.text_source,
        args.article_contents_path,
        args.article_summaries_path,
        args.article_summaries_no_place_path,
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

    existing = {}
    if os.path.exists(out_pred):
        with open(out_pred) as f:
            existing = json.load(f)

    fp_current = prediction_fingerprint(
        args.task,
        args.text_source,
        args.model_path,
        args.seed,
        args.temperature,
        allowed_labels,
    )

    classifier = LLMClassifier(
        model_path=args.model_path, seed=args.seed, temperature=args.temperature
    )

    for pageid in eligible:
        record = existing.get(pageid)
        if should_skip_record(record, fp_current, retry_failed=args.retry_failed):
            continue
        article = next((a for a in articles if str(a.get("pageid")) == pageid), None)
        title = article.get("title", pageid) if article else pageid
        text = text_records[pageid]
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
        record = build_prediction_record(
            pageid=pageid,
            title=title,
            target=target[pageid],
            result=result,
            fingerprint=fp_current,
            extra_metadata=extra_metadata,
        )
        existing[pageid] = record
        write_json_atomic(out_pred, existing, indent=2, ensure_ascii=False)
        status = result.get("parse_status", "error")
        logger.info(f"[{len(existing)}/{len(eligible)}] {pageid} ({title}): {status}")

    y_pred = {
        k: v["prediction"]
        for k, v in existing.items()
        if k in eligible and v.get("parse_status") == "ok"
    }
    y_true = {k: target[k] for k in eligible}
    metrics, _ = compute_metrics(args.task, args.text_source, y_true, y_pred, allowed_labels)
    write_json_atomic(out_metrics, metrics, indent=2, ensure_ascii=False)
    logger.info(f"Wrote metrics to {out_metrics}")


if __name__ == "__main__":
    main()
