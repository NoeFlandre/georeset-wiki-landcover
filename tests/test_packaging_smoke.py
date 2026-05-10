import importlib
from pathlib import Path


def test_dockerfile_includes_runtime_cli_directories():
    dockerfile = Path("Dockerfile").read_text()

    assert "COPY src ./src" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert "COPY tests ./tests" in dockerfile
    assert "uv sync --frozen --all-groups" in dockerfile


def test_documented_cli_modules_are_importable():
    for module_name in [
        "scripts.data.classify_articles",
        "scripts.data.compute_corine_spatial_confidence",
        "scripts.data.summarize_articles",
        "scripts.analysis.evaluate_predictions_with_spatial_confidence",
        "scripts.analysis.summarize_classification_experiment",
    ]:
        assert importlib.import_module(module_name)


def test_cluster_classification_submit_does_not_autosync_by_default():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'AUTO_SYNC="${GEORESET_AUTO_SYNC:-0}"' in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "Auto-sync disabled to avoid repeated SSH polling" in script


def test_classification_sync_interval_defaults_to_admin_safe_value_and_supports_once():
    script = Path("scripts/cluster/sync_classification.sh").read_text()

    assert 'INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"' in script
    assert 'SYNC_ONCE="${SYNC_ONCE:-0}"' in script
    assert 'if [ "${SYNC_ONCE}" = "1" ]; then' in script
