import importlib
import os
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
        Path("src/georeset_wiki_landcover/fetchers/article_summarizer.py"),
        Path("src/georeset_wiki_landcover/fetchers/wiki_content_fetcher.py"),
    ]

    for path in modules:
        text = path.read_text(encoding="utf-8")
        assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"
        assert "read_json_file(" in text


def test_analysis_and_visualization_json_reads_use_read_json_file_helper():
    modules = [
        Path("src/georeset_wiki_landcover/cli/analysis/run_corine_analysis.py"),
        Path("src/georeset_wiki_landcover/visualization/map_visualizer.py"),
        Path("src/georeset_wiki_landcover/fetchers/wiki_fetcher.py"),
    ]

    for path in modules:
        text = path.read_text(encoding="utf-8")
        assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"
        assert "read_json_file(" in text


def test_classification_runner_file_reads_use_read_json_file_helper():
    path = Path("src/georeset_wiki_landcover/classification/runner.py")
    text = path.read_text(encoding="utf-8")
    assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"
    assert "read_json_file(" in text


def test_experiment_cli_modules_use_read_json_file_helper():
    experiment_cli_modules = [
        Path("src/georeset_wiki_landcover/cli/data/compute_corine_spatial_confidence.py"),
        Path("src/georeset_wiki_landcover/cli/analysis/summarize_classification_experiment.py"),
        Path(
            "src/georeset_wiki_landcover/cli/analysis/evaluate_predictions_with_spatial_confidence.py"
        ),
    ]

    for path in experiment_cli_modules:
        text = path.read_text(encoding="utf-8")
        assert "json.load(" not in text, f"{path} must use read_json_file() instead of json.load()"

    direct_json_modules = [
        Path("src/georeset_wiki_landcover/cli/data/compute_corine_spatial_confidence.py"),
        Path("src/georeset_wiki_landcover/cli/analysis/summarize_classification_experiment.py"),
        Path("src/georeset_wiki_landcover/analysis/prediction_loading.py"),
    ]
    for path in direct_json_modules:
        text = path.read_text(encoding="utf-8")
        assert "read_json_file(" in text, f"{path} must import and use read_json_file()"

    spatial_confidence_module = Path(
        "src/georeset_wiki_landcover/cli/data/compute_corine_spatial_confidence.py"
    )
    spatial_confidence_text = spatial_confidence_module.read_text(encoding="utf-8")
    assert "def _load_json(" not in spatial_confidence_text, (
        f"{spatial_confidence_module} should use read_json_file() directly"
    )

    spatial_predictions_module = Path(
        "src/georeset_wiki_landcover/cli/analysis/evaluate_predictions_with_spatial_confidence.py"
    )
    spatial_text = spatial_predictions_module.read_text(encoding="utf-8")
    assert (
        "from georeset_wiki_landcover.analysis.prediction_loading import load_prediction_records"
        in spatial_text
    ), (
        f"{spatial_predictions_module} must use load_prediction_records from georeset_wiki_landcover.analysis.prediction_loading"
    )
    assert "load_prediction_records(" in spatial_text, (
        f"{spatial_predictions_module} must invoke load_prediction_records()"
    )


def test_evidence_evaluators_call_shared_quality_subset_helper_directly():
    modules = [
        Path("src/georeset_wiki_landcover/cli/analysis/evaluate_evidence_card_experiment.py"),
        Path("src/georeset_wiki_landcover/cli/analysis/evaluate_evidence_highlights_experiment.py"),
    ]

    for path in modules:
        text = path.read_text(encoding="utf-8")
        assert "def _subset_masks(" not in text, (
            f"{path} should call quality_subset_masks() directly"
        )
        assert "quality_subset_masks(" in text


def test_pre_commit_scopes_match_ci_quality_commands():
    config = Path(".pre-commit-config.yaml").read_text()

    assert "entry: uv run ruff check ." in config
    assert "entry: uv run ruff format --check ." in config
    assert "entry: uv run mypy src scripts" in config
    assert "pass_filenames: false" in config


def test_mypy_has_strict_overrides_for_typed_core_modules():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert (
        'module = ["georeset_wiki_landcover.utils.*", "georeset_wiki_landcover.config", "georeset_wiki_landcover.contracts"]'
        in pyproject
    )
    assert 'module = ["georeset_wiki_landcover.classification.*"]' in pyproject
    assert (
        'module = ["georeset_wiki_landcover.fetchers.*", "georeset_wiki_landcover.visualization.*", "scripts.*"]'
        in pyproject
    )
    assert "strict = true" in pyproject


def test_package_declares_inline_typing_support():
    assert Path("src/georeset_wiki_landcover/py.typed").is_file()


def test_data_fetcher_tests_do_not_depend_on_local_data_directory():
    text = Path("tests/fetchers/test_data_fetcher.py").read_text(encoding="utf-8")

    assert "pytest.mark.skipif" not in text
    assert "requires_data" not in text
    assert "DATA_FILE" not in text
    assert "data/corine/" not in text


def test_data_fetcher_uses_explicit_cache_invariant_errors():
    text = Path("src/georeset_wiki_landcover/fetchers/data_fetcher.py").read_text(encoding="utf-8")

    assert "assert self.gdf is not None" not in text
    assert 'raise RuntimeError("Dataset could not be loaded")' in text


def test_data_fetcher_path_annotations_accept_pathlike_values():
    from georeset_wiki_landcover.fetchers.data_fetcher import DataFetcher

    init_hints = get_type_hints(DataFetcher.__init__)
    save_hints = get_type_hints(DataFetcher.save_bounds)

    expected_path_type = str | os.PathLike[str]
    assert init_hints["data_path"] == expected_path_type
    assert save_hints["output_path"] == expected_path_type


def test_data_fetcher_tests_exercise_pathlike_save_bounds_contract():
    text = Path("tests/fetchers/test_data_fetcher.py").read_text(encoding="utf-8")

    assert "save_bounds(str(" not in text


def test_data_fetcher_reuses_single_datapaths_default_instance():
    text = Path("src/georeset_wiki_landcover/fetchers/data_fetcher.py").read_text(encoding="utf-8")

    assert text.count("DataPaths()") == 1
    assert "_DATA_PATHS = DataPaths()" in text


def test_data_fetcher_main_log_message_spells_successfully_correctly():
    text = Path("src/georeset_wiki_landcover/fetchers/data_fetcher.py").read_text(encoding="utf-8")

    assert "Succesfully" not in text
    assert "Successfully sampled polygons:" in text


def test_data_fetcher_main_handles_only_known_failures():
    text = Path("src/georeset_wiki_landcover/fetchers/data_fetcher.py").read_text(encoding="utf-8")

    assert "except Exception" not in text
    assert "except (FileNotFoundError, ValueError, RuntimeError)" in text


def test_wiki_article_type_fetcher_uses_explicit_retry_exhaustion_error():
    text = Path("src/georeset_wiki_landcover/fetchers/wiki_article_type_fetcher.py").read_text(
        encoding="utf-8"
    )

    assert "assert last_error is not None" not in text
    assert 'raise RuntimeError("Metadata fetch failed without an underlying exception")' in text


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
    allowed_files = {Path("src/georeset_wiki_landcover/utils/json_io.py")}

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
        "georeset_wiki_landcover.cli.data.classify_articles",
        "georeset_wiki_landcover.cli.data.compute_corine_spatial_confidence",
        "georeset_wiki_landcover.cli.data.filter_pipeline",
        "georeset_wiki_landcover.cli.data.summarize_landuse_evidence",
        "georeset_wiki_landcover.cli.data.summarize_articles",
        "georeset_wiki_landcover.cli.analysis.evaluate_predictions_with_spatial_confidence",
        "georeset_wiki_landcover.cli.analysis.run_corine_analysis",
        "georeset_wiki_landcover.cli.analysis.summarize_classification_experiment",
        "georeset_wiki_landcover.cli.dev.snapshot",
    ]:
        assert importlib.import_module(module_name)


def test_major_codebase_subfolders_have_local_readmes():
    expected_readmes = [
        Path("src/README.md"),
        Path("src/georeset_wiki_landcover/analysis/README.md"),
        Path("src/georeset_wiki_landcover/classification/README.md"),
        Path("src/georeset_wiki_landcover/cli/README.md"),
        Path("src/georeset_wiki_landcover/cli/analysis/README.md"),
        Path("src/georeset_wiki_landcover/cli/data/README.md"),
        Path("src/georeset_wiki_landcover/cli/dev/README.md"),
        Path("src/georeset_wiki_landcover/fetchers/README.md"),
        Path("src/georeset_wiki_landcover/llm/README.md"),
        Path("src/georeset_wiki_landcover/spatial/README.md"),
        Path("src/georeset_wiki_landcover/text/README.md"),
        Path("src/georeset_wiki_landcover/utils/README.md"),
        Path("src/georeset_wiki_landcover/visualization/README.md"),
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
        Path("tests/text/README.md"),
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
        Path("scripts/data/summarize_landuse_evidence.py"),
        Path("scripts/data/summarize_articles.py"),
        Path("scripts/analysis/evaluate_predictions_with_spatial_confidence.py"),
        Path("scripts/analysis/run_corine_analysis.py"),
        Path("scripts/analysis/summarize_classification_experiment.py"),
        Path("scripts/dev/snapshot.py"),
    ]
    for path in wrapper_paths:
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 12
        assert "from georeset_wiki_landcover.cli." in text
        assert "if __name__" in text


def test_packaged_cli_entry_points_are_declared():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject
    for command in [
        "georeset-wiki-landcover-classify-articles",
        "georeset-wiki-landcover-compute-corine-spatial-confidence",
        "georeset-wiki-landcover-filter-pipeline",
        "georeset-wiki-landcover-run-corine-analysis",
        "georeset-wiki-landcover-snapshot",
        "georeset-wiki-landcover-summarize-landuse-evidence",
        "georeset-wiki-landcover-summarize-articles",
        "georeset-wiki-landcover-summarize-classification-experiment",
        "georeset-wiki-landcover-evaluate-spatial-confidence",
    ]:
        assert command in pyproject


def test_georeset_package_imports_work_without_src_compatibility_shims():
    assert importlib.import_module("georeset_wiki_landcover.config")
    assert importlib.import_module("georeset_wiki_landcover.classification.runner")

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

    for root in [Path("src/georeset_wiki_landcover"), Path("scripts"), Path("tests")]:
        for path in root.rglob("*.py"):
            if path in allowed_files:
                continue
            text = path.read_text(encoding="utf-8")
            assert not forbidden.search(text), f"{path} imports historical src namespace"


def test_ruff_first_party_namespace_is_georeset_not_src():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'known-first-party = ["georeset_wiki_landcover", "scripts"]' in pyproject
    assert 'known-first-party = ["src", "scripts"]' not in pyproject


def test_metric_contracts_are_split_by_metric_family():
    contracts = importlib.import_module("georeset_wiki_landcover.contracts")
    metrics_module = importlib.import_module("georeset_wiki_landcover.classification.metrics")
    analysis_metrics = importlib.import_module(
        "georeset_wiki_landcover.analysis.evaluation_metrics"
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
        get_type_hints(analysis_metrics.compute_single_label_subset_metrics)["return"].__args__[0]
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

    assert 'AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"' in script
    assert 'if [ "${AUTO_SYNC}" != "1" ]; then' in script
    assert "Auto-sync disabled to avoid repeated SSH polling" in script


def test_cluster_classification_submit_avoids_sed_generated_wrappers():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert "sed -i" not in script
    assert "wrapper_${TASK}_${TEXT_SOURCE}.sh" not in script
    assert "G5K_REMOTE_HOME" in script
    assert "G5K_REMOTE_PROJECT_DIR" in script
    assert 'OAR_QUEUE="${G5K_OAR_QUEUE:-production}"' in script
    assert 'oarsub -q \\"${OAR_QUEUE}\\"' in script
    assert "env GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK=" in script


def test_cluster_classification_submit_allows_oar_property_override():
    script = Path("scripts/cluster/submit_classification.sh").read_text()

    assert 'OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-gpu_mem>=32000}"' in script
    assert '-p \\"${OAR_PROPERTIES}\\"' in script


def test_classification_sync_interval_defaults_to_admin_safe_value_and_supports_once():
    script = Path("scripts/cluster/sync_classification.sh").read_text()

    assert (
        'OUTPUT_DIR="${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR:-data/classification/runs/default}"'
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
    assert 'read -r -a EXTRA_ARGS <<< "${GEORESET_WIKI_LANDCOVER_EXTRA_ARGS}"' in script
    assert '"${EXTRA_ARGS[@]}"' in script


def test_classification_job_uses_llm_dependency_group_not_manual_pip_install():
    script = Path("scripts/cluster/run_classification_job.sh").read_text()

    assert "uv sync --group dev --group llm" in script
    assert "uv sync --all-groups" not in script
    assert "uv pip install --no-cache-dir huggingface_hub llama-cpp-python" not in script
