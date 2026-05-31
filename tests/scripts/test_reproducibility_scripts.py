import json
from pathlib import Path

from scripts.reproduce_small import run_small_reproduction
from scripts.validate_artifacts import validate_artifacts


def test_small_reproduction_writes_valid_manifested_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"

    manifest = run_small_reproduction(output_dir=output_dir, clean=True)

    assert manifest["mode"] == "small"
    assert manifest["expected_counts"]["wiki_articles"] == 2
    assert manifest["expected_counts"]["corine_level2_summary_predictions"] == 2
    assert manifest["expected_counts"]["osm_summary_predictions"] == 2
    assert validate_artifacts(output_dir, profile="small") == []


def test_small_artifact_validator_reports_stale_manifest_hash(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    summaries_path = output_dir / "data/wiki/article_summaries.json"
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    summaries["100"]["summary"] = "Changed after manifest creation."
    summaries_path.write_text(json.dumps(summaries), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert any(
        "stale artifact hash for data/wiki/article_summaries.json" in violation
        for violation in violations
    )


def test_small_artifact_validator_reports_duplicate_wiki_pageids(tmp_path: Path) -> None:
    output_dir = tmp_path / "small"
    run_small_reproduction(output_dir=output_dir, clean=True)
    wiki_path = output_dir / "data/wiki/wiki_articles.json"
    articles = json.loads(wiki_path.read_text(encoding="utf-8"))
    articles.append(dict(articles[0]))
    wiki_path.write_text(json.dumps(articles), encoding="utf-8")

    violations = validate_artifacts(output_dir, profile="small")

    assert "duplicate wiki pageids: 100" in violations
