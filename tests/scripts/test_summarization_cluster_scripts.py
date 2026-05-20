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
    assert 'AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"' in script
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

    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_INPUT_PATH" in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_OUTPUT_PATH" in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_SEED" in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_TEMPERATURE" in script
    assert "uv run georeset-wiki-landcover-summarize-landuse-evidence" in script
    assert '--input-path "${GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_INPUT_PATH}"' in script
    assert '--output-path "${GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_OUTPUT_PATH}"' in script
    assert "data/wiki/article_landuse_evidence_summaries.json" in script


def test_submit_landuse_evidence_script_uses_safe_default_sync_path():
    script = Path("scripts/cluster/submit_landuse_evidence_summarization.sh").read_text()

    assert (
        'OUTPUT_PATH="${GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_OUTPUT_PATH:-data/wiki/article_landuse_evidence_summaries.json}"'
        in script
    )
    assert 'AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"' in script
    assert 'WALLTIME="${G5K_LANDUSE_EVIDENCE_WALLTIME:-20:00:00}"' in script
    assert "oarsub -q production -l host=1/gpu=1,walltime=${WALLTIME}" in script
    assert "walltime=2:00:00" not in script
    assert 'OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-${OAR_PROPERTIES:-gpu_mem>=32000}}"' in script
    assert '-p \\"${OAR_PROPERTIES}\\"' in script
    assert '-O OAR_%jobid%.out -E OAR_%jobid%.err \\"env' in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_OUTPUT_PATH=" in script
    assert "${OUTPUT_PATH}" in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_TEMPERATURE=" in script
    assert "bash ./" in script
    assert "${JOB_SCRIPT}" in script
    assert "-S bash ./" not in script
    assert "GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK=" not in script
    assert "GEORESET_WIKI_LANDCOVER_MODEL_PATH=" in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "Auto-sync disabled to avoid repeated SSH polling" in script
    assert "Manual sync: GEORESET_WIKI_LANDCOVER_SUMMARY_OUTPUT=${OUTPUT_PATH}" in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_INPUT_PATH" in script
    assert "GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_TEMPERATURE" in script


def test_classification_job_uses_job_local_caches_to_protect_home_quota():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert (
        'JOB_CACHE_DIR="${GEORESET_WIKI_LANDCOVER_JOB_CACHE_DIR:-${TMPDIR:-/tmp}/georeset_${OAR_JOB_ID:-manual}}"'
        in script
    )
    assert 'export HF_HOME="${HF_HOME:-${JOB_CACHE_DIR}/hf}"' in script
    assert 'export UV_CACHE_DIR="${UV_CACHE_DIR:-${JOB_CACHE_DIR}/uv}"' in script
    assert 'mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}"' in script


def test_clip_linear_probe_job_uses_vision_group_and_job_local_caches():
    script = Path("scripts/cluster/run_clip_linear_probe_job.sh").read_text()

    assert "uv sync --group dev --group vision" in script
    assert 'export HF_HOME="${HF_HOME:-${JOB_CACHE_DIR}/hf}"' in script
    assert 'export UV_CACHE_DIR="${UV_CACHE_DIR:-${JOB_CACHE_DIR}/uv}"' in script
    assert "georeset-wiki-landcover-build-clip-label-splits" in script
    assert "georeset-wiki-landcover-fetch-sentinel-patches" in script
    assert "georeset-wiki-landcover-embed-clip-patches" in script
    assert "georeset-wiki-landcover-run-clip-linear-probe-experiment" in script
    assert "georeset-wiki-landcover-run-clip-zero-shot-experiment" in script


def test_submit_clip_linear_probe_syncs_outputs_safely():
    script = Path("scripts/cluster/submit_clip_linear_probe.sh").read_text()

    assert (
        'OUTPUT_DIR="${CLIP_OUTPUT_DIR:-data/experiments/012_clip_linear_probe_weak_labels/'
        'clip_linear_probe_weak_labels_v1}"' in script
    )
    assert 'AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"' in script
    assert "run_clip_linear_probe_job.sh" in script
    assert "uv.lock" not in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "rsync -az" in script
    assert "sentinel_patches_rgb.npz" in script
    assert "clip_embeddings.npz" in script
    assert "zero_shot_clip_metrics.csv" in script


def test_quality_weighted_image_probe_cluster_scripts_are_staged_for_grid5000():
    run_script = Path("scripts/cluster/run_quality_weighted_image_probe_job.sh").read_text()
    submit_script = Path("scripts/cluster/submit_quality_weighted_image_probe.sh").read_text()

    assert "uv sync --group dev --group vision" in run_script
    assert 'WINDOWS="${IMAGE_PROBE_WINDOWS:-320,2240}"' in run_script
    assert 'ENCODERS="${IMAGE_PROBE_ENCODERS:-clip_base}"' in run_script
    assert 'RUN_CONTROLS="${IMAGE_PROBE_RUN_CONTROLS:-0}"' in run_script
    assert (
        'STOP_AFTER_PATCH_VALIDATION="${IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION:-0}"' in run_script
    )
    assert "georeset-wiki-landcover-fetch-sentinel-multiscale-patches" in run_script
    assert "IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION=1" in run_script
    assert "georeset-wiki-landcover-run-quality-weighted-image-zero-shot" in run_script
    assert "georeset-wiki-landcover-run-quality-weighted-image-probe" in run_script
    assert "georeset-wiki-landcover-evaluate-image-probe-training-policy-controls" in run_script
    assert "oarsub" in submit_script
    assert "IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION" in submit_script
    assert "rsync" in submit_script
    assert "014_quality_weighted_multiscale_image_probe" in submit_script
