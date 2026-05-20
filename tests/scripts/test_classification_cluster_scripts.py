from pathlib import Path


def test_run_classification_job_uses_output_dir_with_default():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert (
        'export GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR="${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR:-data/classification/runs/default}"'
        in script
    )
    assert '--output-dir "${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR}"' in script
    assert "georeset-wiki-landcover-classify-articles" in script
    assert "MODEL_REPO_ARGS" in script


def test_submit_classification_sync_command_uses_classification_output_dir_env():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert "GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR=${OUTPUT_DIR}" in script
    assert (
        "Manual sync: GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK=${TASK} "
        "GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE=${TEXT_SOURCE} "
        "GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR=${OUTPUT_DIR} SYNC_ONCE=1 "
        "bash scripts/cluster/sync_classification.sh" in script
    )


def test_submit_classification_uses_configurable_oar_queue_default_and_override():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'OAR_QUEUE="${G5K_OAR_QUEUE:-production}"' in script
    assert 'oarsub -q \\"${OAR_QUEUE}\\"' in script


def test_submit_classification_validates_oar_queue_chars():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'case "${OAR_QUEUE}" in' in script
    assert "*[!A-Za-z0-9_.:-]*" in script
    assert "Invalid OAR queue: ${OAR_QUEUE}." in script


def test_submit_classification_supports_configurable_oar_types():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'OAR_TYPES="${G5K_OAR_TYPES:-}"' in script
    assert 'OAR_TYPE_FLAGS=""' in script
    assert 'if [ -n "${OAR_TYPES}" ]; then' in script
    assert "for OAR_TYPE in ${OAR_TYPES}; do" in script
    assert 'case "${OAR_TYPE}" in' in script
    assert "*[!A-Za-z0-9_.:-]*" in script
    assert "Invalid OAR type: ${OAR_TYPE}." in script
    assert 'OAR_TYPE_FLAGS="${OAR_TYPE_FLAGS} -t ${OAR_TYPE}"' in script
    assert "oarsub -q \\" in script
    assert 'oarsub -q \\"${OAR_QUEUE}\\"' in script
    assert "${OAR_TYPE_FLAGS}" in script
    assert "${OAR_TYPE_ARGS[@]}" not in script
    assert "${OAR_TYPE_FLAGS} \\" in script
