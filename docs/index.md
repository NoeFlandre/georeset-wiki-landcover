# Documentation Index

This directory contains project documentation that is tracked in Git. Generated
data artifacts and experiment tables live under `data/` and are synced through
the Hugging Face bucket instead of Git.

## Main References

- [`../README.md`](../README.md): setup, data sync, packaged CLI commands,
  Docker, Grid5000 launchers, and publishing workflow.
- [`../AGENTS.md`](../AGENTS.md): concise coding-agent operating rules for this
  repository.
- [`../src/README.md`](../src/README.md): package/module overview and local
  quality gates.
- [`agent_playbook.md`](agent_playbook.md): longer workflow guidance for future
  coding-agent runs.
- [`repo_map.md`](repo_map.md): quick path-by-path navigation map.
- [`common_failure_modes.md`](common_failure_modes.md): agent-facing checklist
  of mistakes that can break scientific or workflow assumptions.
- [`architecture.md`](architecture.md): package boundaries, important modules,
  workflow layers, guardrails, tests, and safe extension rules.
- [`data_flow.md`](data_flow.md): concrete input-to-output flow for acquisition,
  filtering, classification, analysis, vision, and cache behavior.
- [`cli.md`](cli.md): packaged entry points, repository scripts, and which
  commands were verified as smoke paths.
- [`reproducibility.md`](reproducibility.md): clean-clone setup, synthetic smoke
  reproduction, full-pipeline commands, validation, stale-cache checks, runtime
  expectations, and troubleshooting.
- [`artifacts.md`](artifacts.md): expected `data/` and synthetic artifact
  structure, manifests, prediction metadata, and validation profiles.
- [`configuration.md`](configuration.md): dependency groups, environment
  variables, CLI flags to record, and default configuration locations.
- [`troubleshooting.md`](troubleshooting.md): common failure modes and recovery
  checks derived from current validators, tests, and workflow code.

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
- [`experiments/005_landuse_evidence_summary/analysis.md`](experiments/005_landuse_evidence_summary/analysis.md):
  verified land-use evidence summary experiment for Qwen and Gemma.
- [`experiments/006_relevance_stratified_evaluation/analysis.md`](experiments/006_relevance_stratified_evaluation/analysis.md):
  analysis-only relevance-stratified evaluation showing how evidence metadata
  filters the frozen Qwen and Gemma predictions.

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
