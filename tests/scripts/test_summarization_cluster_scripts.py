from pathlib import Path


def test_standard_summarization_job_uses_place_mode():
    script = Path("scripts/cluster/run_summarization_job.sh").read_text()

    assert "--output-path data/wiki/article_summaries.json" in script
    assert "--summary-mode place" in script
    assert "uv sync --group dev --group llm" in script
    assert "uv sync --all-groups" not in script
    assert "uv pip install --no-cache-dir huggingface_hub llama-cpp-python" not in script


def test_no_place_summarization_job_uses_no_place_mode():
    script = Path("scripts/cluster/run_summarization_no_place.sh").read_text()

    assert "--output-path data/wiki/article_summaries_no_place.json" in script
    assert "--summary-mode no_place" in script
    assert "uv sync --group dev --group llm" in script
    assert "uv sync --all-groups" not in script
    assert "uv pip install --no-cache-dir huggingface_hub llama-cpp-python" not in script
