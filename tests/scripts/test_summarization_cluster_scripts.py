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


def test_run_landuse_evidence_job_uses_landuse_output_and_cli():
    script = Path("scripts/cluster/run_landuse_evidence_summarization_job.sh").read_text()

    assert "GEORESET_LANDUSE_EVIDENCE_INPUT_PATH" in script
    assert "GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH" in script
    assert "GEORESET_LANDUSE_EVIDENCE_SEED" in script
    assert "GEORESET_LANDUSE_EVIDENCE_TEMPERATURE" in script
    assert "uv run georeset-summarize-landuse-evidence" in script
    assert '--input-path "${GEORESET_LANDUSE_EVIDENCE_INPUT_PATH}"' in script
    assert '--output-path "${GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH}"' in script
    assert "data/wiki/article_landuse_evidence_summaries.json" in script


def test_submit_landuse_evidence_script_uses_safe_default_sync_path():
    script = Path("scripts/cluster/submit_landuse_evidence_summarization.sh").read_text()

    assert (
        'OUTPUT_PATH="${GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH:-data/wiki/article_landuse_evidence_summaries.json}"'
        in script
    )
    assert 'AUTO_SYNC="${GEORESET_AUTO_SYNC:-0}"' in script
    assert 'WALLTIME="${G5K_LANDUSE_EVIDENCE_WALLTIME:-20:00:00}"' in script
    assert "oarsub -q production -l host=1/gpu=1,walltime=${WALLTIME}" in script
    assert "walltime=2:00:00" not in script
    assert 'OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-${OAR_PROPERTIES:-gpu_mem>=32000}}"' in script
    assert '-p \\"${OAR_PROPERTIES}\\"' in script
    assert '-O OAR_%jobid%.out -E OAR_%jobid%.err \\"env' in script
    assert "GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH=" in script
    assert "${OUTPUT_PATH}" in script
    assert "GEORESET_LANDUSE_EVIDENCE_TEMPERATURE=" in script
    assert "bash ./" in script
    assert "${JOB_SCRIPT}" in script
    assert "-S bash ./" not in script
    assert "GEORESET_CLASSIFICATION_TASK=" not in script
    assert "GEORESET_MODEL_PATH=" in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "Auto-sync disabled to avoid repeated SSH polling" in script
    assert "Manual sync: GEORESET_SUMMARY_OUTPUT=${OUTPUT_PATH}" in script
    assert "GEORESET_LANDUSE_EVIDENCE_INPUT_PATH" in script
    assert "GEORESET_LANDUSE_EVIDENCE_TEMPERATURE" in script


def test_classification_job_uses_job_local_caches_to_protect_home_quota():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert 'JOB_CACHE_DIR="${GEORESET_JOB_CACHE_DIR:-${TMPDIR:-/tmp}/georeset_${OAR_JOB_ID:-manual}}"' in script
    assert 'export HF_HOME="${HF_HOME:-${JOB_CACHE_DIR}/hf}"' in script
    assert 'export UV_CACHE_DIR="${UV_CACHE_DIR:-${JOB_CACHE_DIR}/uv}"' in script
    assert 'mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}"' in script
