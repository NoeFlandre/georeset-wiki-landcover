import importlib
import re
from pathlib import Path


def test_dockerfile_includes_runtime_cli_directories():
    dockerfile = Path("Dockerfile").read_text()

    assert "COPY src ./src" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert "COPY tests ./tests" in dockerfile
    assert "uv sync --frozen --group dev --no-install-project" in dockerfile
    assert "uv sync --frozen --group dev" in dockerfile
    assert "uv sync --frozen --all-groups" not in dockerfile


def test_ci_uses_dev_dependency_group_without_llm_group():
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "uv sync --group dev" in workflow
    assert "uv sync --all-groups" not in workflow


def test_llm_dependency_group_declares_manual_cluster_runtime_deps():
    pyproject = Path("pyproject.toml").read_text()

    assert "llm = [" in pyproject
    assert '"huggingface-hub' in pyproject
    assert '"llama-cpp-python' in pyproject


def test_dev_dependency_group_includes_pre_commit():
    pyproject = Path("pyproject.toml").read_text()

    assert '"pre-commit' in pyproject


def test_pre_commit_scopes_match_ci_quality_commands():
    config = Path(".pre-commit-config.yaml").read_text()

    assert "entry: uv run ruff check ." in config
    assert "entry: uv run ruff format --check ." in config
    assert "entry: uv run mypy src scripts" in config
    assert "pass_filenames: false" in config


def test_production_outputs_use_atomic_file_helpers():
    forbidden_patterns = [
        re.compile(r"with open\([^\\n]+,\\s*[\"']w"),
        re.compile(r"\.write_text\("),
        re.compile(r"json\.dump\("),
        re.compile(r"\.to_csv\("),
    ]
    allowed_files = {Path("src/utils/json_io.py")}

    for root in [Path("src"), Path("scripts")]:
        for path in root.rglob("*.py"):
            if path in allowed_files:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                assert not pattern.search(text), (
                    f"{path} still has direct write pattern {pattern.pattern}"
                )


def test_documented_cli_modules_are_importable():
    for module_name in [
        "scripts.data.classify_articles",
        "scripts.data.compute_corine_spatial_confidence",
        "scripts.data.summarize_articles",
        "scripts.analysis.evaluate_predictions_with_spatial_confidence",
        "scripts.analysis.summarize_classification_experiment",
    ]:
        assert importlib.import_module(module_name)


def test_python_scripts_do_not_mutate_sys_path_for_imports():
    for path in Path("scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in text
        assert "sys.path.append" not in text


def test_cluster_classification_submit_does_not_autosync_by_default():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'AUTO_SYNC="${GEORESET_AUTO_SYNC:-0}"' in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "Auto-sync disabled to avoid repeated SSH polling" in script


def test_cluster_classification_submit_avoids_sed_generated_wrappers():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert "sed -i" not in script
    assert "wrapper_${TASK}_${TEXT_SOURCE}.sh" not in script
    assert "G5K_REMOTE_HOME" in script
    assert "G5K_REMOTE_PROJECT_DIR" in script


def test_classification_sync_interval_defaults_to_admin_safe_value_and_supports_once():
    script = Path("scripts/cluster/sync_classification.sh").read_text()

    assert 'INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"' in script
    assert 'SYNC_ONCE="${SYNC_ONCE:-0}"' in script
    assert 'if [ "${SYNC_ONCE}" = "1" ]; then' in script


def test_cluster_scripts_do_not_hardcode_personal_home_path():
    for path in [
        Path("scripts/cluster/run_classification_job.sh"),
        Path("scripts/cluster/submit_classification.sh"),
        Path("scripts/cluster/sync_classification.sh"),
    ]:
        assert "/home/nflandre" not in path.read_text()


def test_classification_job_passes_extra_args_as_array():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert "EXTRA_ARGS=()" in script
    assert 'read -r -a EXTRA_ARGS <<< "${GEORESET_EXTRA_ARGS}"' in script
    assert '"${EXTRA_ARGS[@]}"' in script


def test_classification_job_uses_llm_dependency_group_not_manual_pip_install():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert "uv sync --group dev --group llm" in script
    assert "uv sync --all-groups" not in script
    assert "uv pip install --no-cache-dir huggingface_hub llama-cpp-python" not in script
