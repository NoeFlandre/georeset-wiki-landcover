import json
from pathlib import Path

import pytest

from georeset.cli.data.build_evidence_cards import main


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_evidence_cards_cli_writes_deterministic_records(tmp_path):
    contents_path = tmp_path / "article_contents.json"
    evidence_path = tmp_path / "article_landuse_evidence_summaries.json"
    article_types_path = tmp_path / "article_type_assignments.csv"
    spatial_path = tmp_path / "spatial_confidence.csv"
    quality_path = tmp_path / "quality_scores.csv"
    output_path = tmp_path / "article_evidence_cards.json"
    old_experiment_dir = tmp_path / "old_experiment"
    old_experiment_dir.mkdir()
    sentinel = old_experiment_dir / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    _write_json(
        contents_path,
        {
            "100": {
                "title": "Forêt de Test",
                "content": "Texte complet avec Forêt de Test.",
                "url": "https://example.test/100",
            },
            "200": {"title": "Village", "content": "Texte sans métadonnées."},
        },
    )
    _write_json(
        evidence_path,
        {
            "100": {
                "landcover_relevance": "high",
                "uncertainty": "low",
                "evidence_types": ["forest"],
                "evidence_sentences_no_place": ["Des boisements sont mentionnés."],
                "landuse_evidence_summary": "Le paysage comprend des boisements.",
                "evidence_sentences_count": 1,
                "landuse_evidence_summary_char_count": 38,
            }
        },
    )
    article_types_path.write_text(
        "pageid,title,primary_article_type,candidate_article_types\n"
        "100,Forêt de Test,natural_landscape,\"['natural_landscape']\"\n",
        encoding="utf-8",
    )
    spatial_path.write_text(
        "pageid,point_label_share_250m,point_label_share_500m,dominant_matches_point_label_250m,point_label,dominant_label_250m\n"
        "100,0.8,0.7,true,31,31\n",
        encoding="utf-8",
    )
    quality_path.write_text(
        "pageid,quality_score,quality_bin,recommended_use\n"
        "100,7.1,quality_very_high,use_for_training\n",
        encoding="utf-8",
    )

    main(
        [
            "--article-contents-path",
            str(contents_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--article-type-metadata-path",
            str(article_types_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--quality-scores-path",
            str(quality_path),
            "--output-path",
            str(output_path),
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(output) == {"100", "200"}
    assert output["100"]["pageid"] == "100"
    assert output["100"]["title"] == "Forêt de Test"
    assert "Forêt de Test" not in output["100"]["evidence_card"]
    assert "31" not in output["100"]["evidence_card"]
    assert output["100"]["evidence_sentence_count"] == 1
    assert output["100"]["evidence_card_char_count"] == len(output["100"]["evidence_card"])
    assert output["100"]["metadata"]["source"] == "deterministic_evidence_card"
    assert output["100"]["metadata"]["version"] == 1
    assert (
        output["100"]["metadata"]["text_variants"]["evidence_card"]["uses_raw_content"]
        is False
    )
    assert (
        output["100"]["metadata"]["text_variants"]["content_with_evidence_card"][
            "uses_raw_content"
        ]
        is True
    )
    assert "Aucun indice factuel explicite" in output["200"]["evidence_card"]
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_build_evidence_cards_cli_sanitizes_missing_metadata_for_json(tmp_path):
    contents_path = tmp_path / "article_contents.json"
    evidence_path = tmp_path / "article_landuse_evidence_summaries.json"
    article_types_path = tmp_path / "article_type_assignments.csv"
    spatial_path = tmp_path / "spatial_confidence.csv"
    quality_path = tmp_path / "quality_scores.csv"
    output_path = tmp_path / "article_evidence_cards.json"

    _write_json(
        contents_path,
        {
            "100": {
                "title": "Zone",
                "content": "Texte complet.",
                "url": "https://example.test/100",
            }
        },
    )
    _write_json(
        evidence_path,
        {
            "100": {
                "landcover_relevance": float("nan"),
                "uncertainty": float("nan"),
                "evidence_types": [float("nan"), "forêt"],
            }
        },
    )
    article_types_path.write_text(
        "pageid,primary_article_type,candidate_article_types\n"
        "100,,\n",
        encoding="utf-8",
    )
    spatial_path.write_text(
        "pageid,point_label_share_250m,point_label_share_500m,dominant_matches_point_label_250m\n"
        "100,,,\n",
        encoding="utf-8",
    )
    quality_path.write_text(
        "pageid,quality_score,quality_bin,recommended_use\n"
        "100,,,\n",
        encoding="utf-8",
    )

    main(
        [
            "--article-contents-path",
            str(contents_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--article-type-metadata-path",
            str(article_types_path),
            "--spatial-confidence-path",
            str(spatial_path),
            "--quality-scores-path",
            str(quality_path),
            "--output-path",
            str(output_path),
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))

    record = output["100"]
    assert record["landcover_relevance"] is None
    assert record["uncertainty"] is None
    assert record["evidence_types"] == ["forêt"]
    assert record["candidate_article_types"] == ["other_or_unclear"]
    assert record["point_label_share_250m"] is None
    assert record["point_label_share_500m"] is None
    assert record["dominant_matches_point_label_250m"] is None
    assert record["quality_score"] is None
    assert record["quality_bin"] is None
    assert record["recommended_use"] is None
    assert json.dumps(record, allow_nan=False)


def test_build_evidence_cards_cli_fails_loudly_for_non_mapping_article_contents(
    tmp_path,
):
    contents_path = tmp_path / "article_contents.json"
    output_path = tmp_path / "article_evidence_cards.json"
    _write_json(contents_path, [{"pageid": 100, "content": "not a mapping"}])

    with pytest.raises(ValueError, match="article contents"):
        main(
            [
                "--article-contents-path",
                str(contents_path),
                "--output-path",
                str(output_path),
            ]
        )

    assert not output_path.exists()
