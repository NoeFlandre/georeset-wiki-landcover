"""Smoke test for run_corine_analysis script."""

from scripts.analysis.run_corine_analysis import run


def test_run_corine_analysis_exports_run():
    """Should have a run function."""
    assert callable(run)
