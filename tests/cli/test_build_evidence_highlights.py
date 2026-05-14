import json
from pathlib import Path

from georeset.cli.data.build_evidence_highlights import main


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_evidence_highlights_cli_writes_records(tmp_path):
    contents_path = tmp_path / "contents.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "highlights.json"

    _write_json(
        contents_path,
        {"100": {"title": "Lieu", "content": "Texte complet avec le nom Lieu."}},
    )
    _write_json(
        evidence_path,
        {
            "100": {
                "landcover_relevance": "high",
                "uncertainty": "low",
                "evidence_types": ["forest"],
                "evidence_sentences_no_place": ["Des boisements sont présents."],
            }
        },
    )

    main(
        [
            "--article-contents-path",
            str(contents_path),
            "--evidence-metadata-path",
            str(evidence_path),
            "--output-path",
            str(output_path),
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    record = output["100"]

    assert "Des boisements sont présents." in record["evidence_highlights"]
    assert record["content_with_evidence_highlights"].startswith(record["evidence_highlights"])
    assert record["metadata"]["source"] == "deterministic_evidence_highlights"


def test_build_evidence_highlights_cli_fails_loudly_for_non_mapping_article_contents(
    tmp_path,
):
    contents_path = tmp_path / "contents.json"
    output_path = tmp_path / "highlights.json"
    _write_json(contents_path, [{"pageid": 1}])

    try:
        main(["--article-contents-path", str(contents_path), "--output-path", str(output_path)])
    except ValueError as exc:
        assert "article contents" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert not output_path.exists()
