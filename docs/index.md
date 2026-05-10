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

- [`experiments/article_text_classification_e2e_with_shuffled_control_v1_analysis.md`](experiments/article_text_classification_e2e_with_shuffled_control_v1_analysis.md):
  verified analysis of the first frozen classification batch with shuffled
  controls.
- [`experiments/corine_spatial_confidence_v1.md`](experiments/corine_spatial_confidence_v1.md):
  method note for CORINE buffer-purity diagnostics.
- [`experiments/article_text_classification_spatial_confidence_v1_analysis.md`](experiments/article_text_classification_spatial_confidence_v1_analysis.md):
  verified analysis of spatial-confidence subset reevaluation.

## Design And Brainstorming Notes

- [`brainstorm/gpt5.5-pro.md`](brainstorm/gpt5.5-pro.md): historical external
  research-advice note. Treat it as context, not as current run instructions or
  an authoritative description of the current code layout.

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
