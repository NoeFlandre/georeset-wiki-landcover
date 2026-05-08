"""Classify Wikipedia articles into land-cover labels using an LLM."""

import argparse
import hashlib
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import geopandas as gpd

from src.classification.ground_truth import build_corine_ground_truth, build_osm_ground_truth
from src.classification.labels import corine_level2_labels, osm_allowed_labels
from src.classification.llm_classifier import LLMClassifier
from src.classification.metrics import multilabel_metrics, single_label_metrics
from src.fetchers.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Wikipedia articles into land-cover labels."
    )
    parser.add_argument(
        "--task", choices=["corine_level2", "osm"], default="corine_level2"
    )
    parser.add_argument(
        "--text-source",
        choices=["summary", "summary_no_place", "content"],
        default="summary",
    )
    parser.add_argument("--wiki-articles-path", default="data/wiki/wiki_articles.json")
    parser.add_argument(
        "--article-contents-path", default="data/wiki/article_contents.json"
    )
    parser.add_argument(
        "--article-summaries-path", default="data/wiki/article_summaries.json"
    )
    parser.add_argument(
        "--article-summaries-no-place-path",
        default="data/wiki/article_summaries_no_place.json",
    )
    parser.add_argument(
        "--osm-polygons-path", default="data/osm/osm_project_polygons.geojson"
    )
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
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def load_text_source(
    text_source: str,
    article_contents_path: str,
    article_summaries_path: str,
    article_summaries_no_place_path: str,
) -> dict[str, str]:
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
    y_true: dict,
    y_pred: dict,
    allowed_labels: list[str],
) -> tuple[dict, list[str]]:
    """Returns (metrics_dict, labels_evaluated)."""
    eval_labels = sorted(
        {v
         for vals in y_true.values()
         for v in (vals if isinstance(vals, list) else [vals])}
        | {v
           for vals in y_pred.values()
           for v in (vals if isinstance(vals, list) else [vals])}
    )
    if task == "corine_level2":
        metrics = single_label_metrics(y_true, y_pred, labels=eval_labels)
    else:
        metrics = multilabel_metrics(y_true, y_pred, labels=eval_labels)
    metrics["task"] = task
    metrics["text_source"] = text_source
    metrics["allowed_labels"] = sorted(allowed_labels)
    metrics["labels_evaluated"] = eval_labels
    return metrics, eval_labels


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    out_pred = os.path.join(
        args.output_dir, f"{args.task}_{args.text_source}_predictions.json"
    )
    out_metrics = os.path.join(
        args.output_dir, f"{args.task}_{args.text_source}_metrics.json"
    )

    with open(args.wiki_articles_path) as f:
        articles = json.load(f)

    text_records = load_text_source(
        args.text_source,
        args.article_contents_path,
        args.article_summaries_path,
        args.article_summaries_no_place_path,
    )

    if args.task == "corine_level2":
        fetcher = DataFetcher(args.corine_polygons_path)
        corine_gdf = fetcher.load_data(exclude_artificial=True)
        target = build_corine_ground_truth(articles, corine_gdf)
        allowed_labels = corine_level2_labels(corine_gdf)
        label_descriptions = {
            k: v
            for k, v in {
                "21": "Arable land",
                "22": "Permanent crops",
                "23": "Pastures",
                "24": "Heterogeneous agricultural areas",
                "31": "Forests",
                "32": "Shrub and/or herbaceous vegetation associations",
                "33": "Open spaces with little or no vegetation",
                "41": "Inland wetlands",
                "51": "Inland waters",
            }.items()
            if k in allowed_labels
        }
    else:
        osm_gdf = gpd.read_file(args.osm_polygons_path)
        target = build_osm_ground_truth(articles, osm_gdf)
        allowed_labels = osm_allowed_labels()
        label_descriptions = {}

    eligible = [pid for pid in text_records if pid in target]
    if args.limit:
        eligible = eligible[: args.limit]

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
        if (
            record
            and record.get("parse_status") == "ok"
            and record.get("metadata", {}).get("fingerprint") == fp_current
        ):
            continue
        article = next(
            (a for a in articles if str(a.get("pageid")) == pageid), None
        )
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
        record = {
            "pageid": pageid,
            "title": title,
            "target": target[pageid],
            "prediction": result.get("prediction"),
            "parse_status": result.get("parse_status"),
            "raw_response": result.get("raw_response"),
            "error": result.get("error"),
            "metadata": {**result.get("metadata", {}), "fingerprint": fp_current},
        }
        existing[pageid] = record
        with open(out_pred, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        status = result.get("parse_status", "error")
        logger.info(f"[{len(existing)}/{len(eligible)}] {pageid} ({title}): {status}")

    y_pred = {
        k: v["prediction"]
        for k, v in existing.items()
        if k in eligible and v.get("parse_status") == "ok"
    }
    y_true = {k: target[k] for k in eligible}
    metrics, _ = compute_metrics(
        args.task, args.text_source, y_true, y_pred, allowed_labels
    )
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote metrics to {out_metrics}")


if __name__ == "__main__":
    main()
