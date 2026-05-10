from pathlib import Path


def test_standard_summarization_job_uses_place_mode():
    script = Path("scripts/cluster/run_summarization_job.sh").read_text()

    assert "--output-path data/wiki/article_summaries.json" in script
    assert "--summary-mode place" in script
    assert "uv sync --group dev --group llm" in script
    assert "uv sync --all-groups" not in script
    assert "uv pip install --no-cache-dir huggingface_hub llama-cpp-python" not in script
    assert 'REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${HOME}/${REMOTE_DIR}}"' in script
    assert 'cd "${REMOTE_PROJECT_DIR}"' in script
    assert "/home/nflandre" not in script


def test_no_place_summarization_job_uses_no_place_mode():
    script = Path("scripts/cluster/run_summarization_no_place.sh").read_text()

    assert "--output-path data/wiki/article_summaries_no_place.json" in script
    assert "--summary-mode no_place" in script
    assert "uv sync --group dev --group llm" in script
    assert "uv sync --all-groups" not in script
    assert "uv pip install --no-cache-dir huggingface_hub llama-cpp-python" not in script
    assert 'REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${HOME}/${REMOTE_DIR}}"' in script
    assert 'cd "${REMOTE_PROJECT_DIR}"' in script
    assert "/home/nflandre" not in script


def test_submit_summarization_uses_parameterized_remote_paths_and_safe_sync_default():
    script = Path("scripts/cluster/submit_summarization.sh").read_text()

    assert 'REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"' in script
    assert 'REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"' in script
    assert 'AUTO_SYNC="${GEORESET_AUTO_SYNC:-0}"' in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "${REMOTE_PROJECT_DIR}/OAR_${JOB_ID}.err" in script
    assert "G5K_REMOTE_HOME" in script
    assert "/home/nflandre" not in script


def test_sync_summaries_uses_parameterized_remote_paths_and_once_mode():
    script = Path("scripts/cluster/sync_summaries.sh").read_text()

    assert 'REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"' in script
    assert 'REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"' in script
    assert 'INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"' in script
    assert 'SYNC_ONCE="${SYNC_ONCE:-0}"' in script
    assert 'if [ "${SYNC_ONCE}" = "1" ]; then' in script
    assert "/home/nflandre" not in script
