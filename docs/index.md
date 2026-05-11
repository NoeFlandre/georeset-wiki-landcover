# Documentation Index

This directory contains project documentation that is tracked in Git. Generated
data artifacts and experiment tables live under `data/` and are synced through
the Hugging Face bucket instead of Git.

## Main References

- [`../README.md`](../README.md): setup, data sync, packaged CLI commands,
  Docker, Grid5000 launchers, and publishing workflow.
- [`../src/README.md`](../src/README.md): package/module overview and local
  quality gates.

## Experiments

- [`experiments/README.md`](experiments/README.md): chronological index of
  experiment reports and their `data/experiments/` artifacts.
- [`experiments/001_qwen_e2e_shuffled_control/analysis.md`](experiments/001_qwen_e2e_shuffled_control/analysis.md):
  verified analysis of the first frozen classification batch with shuffled
  controls.
- [`experiments/002_corine_spatial_confidence/README.md`](experiments/002_corine_spatial_confidence/README.md):
  method note for CORINE buffer-purity diagnostics.
- [`experiments/003_qwen_spatial_confidence_reevaluation/analysis.md`](experiments/003_qwen_spatial_confidence_reevaluation/analysis.md):
  verified analysis of spatial-confidence subset reevaluation.
- [`experiments/004_gemma4_model_rerun_and_comparison/analysis.md`](experiments/004_gemma4_model_rerun_and_comparison/analysis.md):
  verified Gemma rerun and Qwen-vs-Gemma comparison.

## Diagrams

- `diagrams/pipeline_diagram.tex`: TikZ source for the README diagram.
- `diagrams/pipeline_diagram.pdf`: compiled PDF.
- `diagrams/pipeline_diagram-1.png`: PNG render used by the main README.

To regenerate the diagram:

```bash
cd docs/diagrams
pdflatex pipeline_diagram.tex
pdftoppm -png -r 150 pipeline_diagram.pdf pipeline_diagram
```
