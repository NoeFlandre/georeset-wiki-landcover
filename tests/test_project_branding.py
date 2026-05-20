import re
from pathlib import Path


def test_distribution_and_import_namespace_use_wiki_landcover_brand() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "georeset-wiki-landcover"' in pyproject
    assert 'packages = ["src/georeset_wiki_landcover"]' in pyproject
    assert Path("src/georeset_wiki_landcover").is_dir()
    assert not Path("src/georeset").exists()


def test_hugging_face_bucket_references_use_wiki_landcover_bucket() -> None:
    expected = "hf://buckets/NoeFlandre/georeset-wiki-landcover"

    checked_text = "\n".join(
        path.read_text(encoding="utf-8") for path in [Path("README.md"), Path("src/README.md")]
    )

    assert expected in checked_text
    assert not re.search(r"hf://buckets/NoeFlandre/georeset(?!-wiki-landcover)", checked_text)
