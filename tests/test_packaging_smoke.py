import importlib
import re
from pathlib import Path
from typing import get_type_hints


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


def test_ci_coverage_scope_matches_pytest_addopts():
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "--cov=src" not in workflow
    assert "uv run pytest -q" in workflow


def test_llm_dependency_group_declares_manual_cluster_runtime_deps():
    pyproject = Path("pyproject.toml").read_text()

    assert "llm = [" in pyproject
    assert '"huggingface-hub' in pyproject
    assert '"llama-cpp-python' in pyproject


def test_dev_dependency_group_includes_pre_commit():
    pyproject = Path("pyproject.toml").read_text()

    assert '"pre-commit' in pyproject


def test_direct_runtime_imports_are_declared_project_dependencies():
    pyproject = Path("pyproject.toml").read_text()

    assert '"typing-extensions' in pyproject


def test_fetcher_file_reads_use_read_json_file_helper():
    modules = [
        Path("src/georeset/fetchers/article_summarizer.py"),
        Path("src/georeset/fetchers/wiki_content_fetcher.py"),
    ]

    for path in modules:
        text = path.read_text(encoding="utf-8")
        assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"
        assert "read_json_file(" in text


def test_classification_runner_file_reads_use_read_json_file_helper():
    path = Path("src/georeset/classification/runner.py")
    text = path.read_text(encoding="utf-8")
    assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"
    assert "read_json_file(" in text


def test_experiment_cli_modules_use_read_json_file_helper():
    modules = [
        Path("src/georeset/cli/data/compute_corine_spatial_confidence.py"),
        Path("src/georeset/cli/analysis/summarize_classification_experiment.py"),
        Path("src/georeset/cli/analysis/evaluate_predictions_with_spatial_confidence.py"),
    ]

    for path in modules:
        text = path.read_text(encoding="utf-8")
        assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"
        assert "read_json_file(" in text, f"{path} must import and use read_json_file()"


def test_pre_commit_scopes_match_ci_quality_commands():
    config = Path(".pre-commit-config.yaml").read_text()

    assert "entry: uv run ruff check ." in config
    assert "entry: uv run ruff format --check ." in config
    assert "entry: uv run mypy src scripts" in config
    assert "pass_filenames: false" in config


def test_mypy_has_strict_overrides_for_typed_core_modules():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'module = ["georeset.utils.*", "georeset.config", "georeset.contracts"]' in pyproject
    assert 'module = ["georeset.classification.*"]' in pyproject
    assert 'module = ["georeset.fetchers.*", "georeset.visualization.*", "scripts.*"]' in pyproject
    assert "strict = true" in pyproject


def test_production_outputs_use_atomic_file_helpers():
    forbidden_patterns = [
        re.compile(r"with open\([^\\n]+,\\s*[\"']w"),
        re.compile(r"\.write_text\("),
        re.compile(r"json\.dump\("),
        re.compile(r"\.to_csv\("),
        re.compile(r"\.to_file\("),
        re.compile(r"\.save\("),
        re.compile(r"\.to_parquet\("),
    ]
    allowed_files = {Path("src/georeset/utils/json_io.py")}

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
        "georeset.cli.data.classify_articles",
        "georeset.cli.data.compute_corine_spatial_confidence",
        "georeset.cli.data.filter_pipeline",
        "georeset.cli.data.summarize_articles",
        "georeset.cli.analysis.evaluate_predictions_with_spatial_confidence",
        "georeset.cli.analysis.run_corine_analysis",
        "georeset.cli.analysis.summarize_classification_experiment",
        "georeset.cli.dev.snapshot",
    ]:
        assert importlib.import_module(module_name)


def test_major_codebase_subfolders_have_local_readmes():
    expected_readmes = [
        Path("src/README.md"),
        Path("src/georeset/analysis/README.md"),
        Path("src/georeset/classification/README.md"),
        Path("src/georeset/cli/README.md"),
        Path("src/georeset/cli/analysis/README.md"),
        Path("src/georeset/cli/data/README.md"),
        Path("src/georeset/cli/dev/README.md"),
        Path("src/georeset/fetchers/README.md"),
        Path("src/georeset/llm/README.md"),
        Path("src/georeset/spatial/README.md"),
        Path("src/georeset/utils/README.md"),
        Path("src/georeset/visualization/README.md"),
        Path("scripts/README.md"),
        Path("scripts/analysis/README.md"),
        Path("scripts/cluster/README.md"),
        Path("scripts/data/README.md"),
        Path("scripts/dev/README.md"),
        Path("tests/README.md"),
        Path("tests/analysis/README.md"),
        Path("tests/classification/README.md"),
        Path("tests/fetchers/README.md"),
        Path("tests/scripts/README.md"),
        Path("tests/spatial/README.md"),
        Path("tests/utils/README.md"),
        Path("tests/visualization/README.md"),
    ]

    for path in expected_readmes:
        assert path.exists(), f"Missing local documentation: {path}"
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) >= 5


def test_repo_scripts_are_thin_wrappers_over_packaged_cli_modules():
    wrapper_paths = [
        Path("scripts/data/classify_articles.py"),
        Path("scripts/data/compute_corine_spatial_confidence.py"),
        Path("scripts/data/filter_pipeline.py"),
        Path("scripts/data/summarize_articles.py"),
        Path("scripts/analysis/evaluate_predictions_with_spatial_confidence.py"),
        Path("scripts/analysis/run_corine_analysis.py"),
        Path("scripts/analysis/summarize_classification_experiment.py"),
        Path("scripts/dev/snapshot.py"),
    ]
    for path in wrapper_paths:
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 12
        assert "from georeset.cli." in text
        assert "if __name__" in text


def test_packaged_cli_entry_points_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    for command in [
        "georeset-classify-articles",
        "georeset-compute-corine-spatial-confidence",
        "georeset-filter-pipeline",
        "georeset-run-corine-analysis",
        "georeset-snapshot",
        "georeset-summarize-articles",
        "georeset-summarize-classification-experiment",
        "georeset-evaluate-spatial-confidence",
    ]:
        assert command in pyproject


def test_georeset_package_imports_work_without_src_compatibility_shims():
    assert importlib.import_module("georeset.config")
    assert importlib.import_module("georeset.classification.runner")

    obsolete_shims = [
        Path("src/__init__.py"),
        Path("src/config.py"),
        Path("src/contracts.py"),
        Path("src/analysis/__init__.py"),
        Path("src/classification/__init__.py"),
        Path("src/fetchers/__init__.py"),
        Path("src/llm/__init__.py"),
        Path("src/spatial/__init__.py"),
        Path("src/utils/__init__.py"),
        Path("src/visualization/__init__.py"),
    ]
    assert not any(path.exists() for path in obsolete_shims)


def test_new_code_does_not_import_historical_src_namespace():
    forbidden = re.compile(r"\b(from src\.|import src\.|src\.)")
    allowed_files = {Path("tests/test_packaging_smoke.py")}

    for root in [Path("src/georeset"), Path("scripts"), Path("tests")]:
        for path in root.rglob("*.py"):
            if path in allowed_files:
                continue
            text = path.read_text(encoding="utf-8")
            assert not forbidden.search(text), f"{path} imports historical src namespace"


def test_ruff_first_party_namespace_is_georeset_not_src():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'known-first-party = ["georeset", "scripts"]' in pyproject
    assert 'known-first-party = ["src", "scripts"]' not in pyproject


def test_metric_contracts_are_split_by_metric_family():
    contracts = importlib.import_module("georeset.contracts")
    metrics_module = importlib.import_module("georeset.classification.metrics")
    spatial_eval = importlib.import_module(
        "georeset.cli.analysis.evaluate_predictions_with_spatial_confidence"
    )

    assert hasattr(contracts, "SingleLabelMetricResult")
    assert hasattr(contracts, "MultiLabelMetricResult")
    assert hasattr(contracts, "SpatialSubsetMetricResult")
    assert not hasattr(contracts, "MetricResult")

    assert (
        get_type_hints(metrics_module.single_label_metrics)["return"]
        is contracts.SingleLabelMetricResult
    )
    assert (
        get_type_hints(metrics_module.multilabel_metrics)["return"]
        is contracts.MultiLabelMetricResult
    )
    assert (
        get_type_hints(spatial_eval._single_metrics)["return"].__args__[0]
        is contracts.SpatialSubsetMetricResult
    )


def test_classify_articles_wrapper_does_not_mutate_runner_globals():
    wrapper = Path("scripts/data/classify_articles.py").read_text(encoding="utf-8")

    assert "_runner).LLMClassifier" not in wrapper
    assert "_runner.LLMClassifier =" not in wrapper


def test_python_scripts_do_not_mutate_sys_path_for_imports():
    for path in Path("scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in text
        assert "sys.path.append" not in text


def test_reusable_script_functions_do_not_print():
    reusable_scripts = [
        Path("scripts/analysis/run_corine_analysis.py"),
        Path("scripts/data/filter_pipeline.py"),
        Path("scripts/analysis/summarize_classification_experiment.py"),
    ]
    for path in reusable_scripts:
        tree = __import__("ast").parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, __import__("ast").FunctionDef):
                continue
            if node.name in {"main"}:
                continue
            calls_print = any(
                isinstance(child, __import__("ast").Call)
                and isinstance(child.func, __import__("ast").Name)
                and child.func.id == "print"
                for child in __import__("ast").walk(node)
            )
            assert not calls_print, f"{path}:{node.name} prints inside reusable function"


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
    assert "oarsub -q production" in script
    assert "env GEORESET_CLASSIFICATION_TASK=" in script


def test_cluster_classification_submit_allows_oar_property_override():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-gpu_mem>=32000}"' in script
    assert '-p \\"${OAR_PROPERTIES}\\"' in script


def test_classification_sync_interval_defaults_to_admin_safe_value_and_supports_once():
    script = Path("scripts/cluster/sync_classification.sh").read_text()

    assert (
        'OUTPUT_DIR="${GEORESET_CLASSIFICATION_OUTPUT_DIR:-data/classification/runs/default}"'
        in script
    )
    assert 'OUTPUT_PREFIX="${OUTPUT_DIR}/${TASK}_${TEXT_SOURCE}"' in script
    assert 'INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"' in script
    assert 'SYNC_ONCE="${SYNC_ONCE:-0}"' in script
    assert 'if [ "${SYNC_ONCE}" = "1" ]; then' in script


def test_cluster_scripts_do_not_hardcode_personal_home_path():
    for path in [
        Path("scripts/cluster/run_classification_job.sh"),
        Path("scripts/cluster/submit_classification.sh"),
        Path("scripts/cluster/sync_classification.sh"),
        Path("scripts/cluster/run_summarization_job.sh"),
        Path("scripts/cluster/run_summarization_no_place.sh"),
        Path("scripts/cluster/submit_summarization.sh"),
        Path("scripts/cluster/sync_summaries.sh"),
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
