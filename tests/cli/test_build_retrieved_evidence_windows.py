import json
from pathlib import Path

from georeset_wiki_landcover.cli.data.build_retrieved_evidence_windows import main


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_build_retrieved_evidence_windows_cli_writes_records(tmp_path: Path) -> None:
    contents_path = tmp_path / "contents.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "retrieved.json"

    _write_json(
        contents_path,
        {"100": {"title": "Lieu", "content": "Intro. Des boisements sont présents. Fin."}},
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
            "--seed",
            "13",
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    record = output["100"]

    assert "Des boisements sont présents." in record["retrieved_evidence_windows"]
    assert record["random_sentence_windows"]
    assert record["metadata"]["source"] == "deterministic_retrieved_evidence_windows"


def test_build_retrieved_evidence_windows_cli_fails_for_non_mapping_articles(
    tmp_path: Path,
) -> None:
    contents_path = tmp_path / "contents.json"
    output_path = tmp_path / "retrieved.json"
    _write_json(contents_path, [{"pageid": 1}])

    try:
        main(["--article-contents-path", str(contents_path), "--output-path", str(output_path)])
    except ValueError as exc:
        assert "article contents" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert not output_path.exists()
