from pathlib import Path


def test_run_classification_job_uses_output_dir_with_default():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert (
        'export GEORESET_CLASSIFICATION_OUTPUT_DIR="${GEORESET_CLASSIFICATION_OUTPUT_DIR:-data/classification/runs/default}"'
        in script
    )
    assert (
        "--output-dir \"${GEORESET_CLASSIFICATION_OUTPUT_DIR}\""
        in script
    )
    assert "georeset-classify-articles" in script
    assert "MODEL_REPO_ARGS" in script


def test_submit_classification_sync_command_uses_classification_output_dir_env():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert "GEORESET_CLASSIFICATION_OUTPUT_DIR=${OUTPUT_DIR}" in script
    assert (
        "Manual sync: GEORESET_CLASSIFICATION_TASK=${TASK} "
        "GEORESET_CLASSIFICATION_TEXT_SOURCE=${TEXT_SOURCE} "
        "GEORESET_CLASSIFICATION_OUTPUT_DIR=${OUTPUT_DIR} SYNC_ONCE=1 "
        "bash scripts/cluster/sync_classification.sh"
        in script
    )
