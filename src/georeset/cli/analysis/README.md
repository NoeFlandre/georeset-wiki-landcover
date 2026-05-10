# Analysis CLIs

This package contains command-line tools that read already-generated artifacts
and produce derived tables, summaries, or reevaluations.

## Commands

- `run_corine_analysis.py`
  - Fetches or loads project-scoped OSM polygons.
  - Computes CORINE class distributions inside those polygons.
  - Writes the OSM/CORINE map, GeoJSON, and distribution CSV.
- `summarize_classification_experiment.py`
  - Reads classification prediction/metrics files from an experiment folder.
  - Regenerates overview tables, shuffled deltas, majority baselines, and
    experiment README files.
- `evaluate_predictions_with_spatial_confidence.py`
  - Joins frozen classification predictions to a CORINE spatial-confidence
    table.
  - Recomputes metrics on fixed spatial subsets without rerunning the LLM.

## Output Discipline

These tools must not mutate frozen parent experiment inputs. Write new derived
artifacts into a separate output directory and include a manifest when creating
an experiment folder.
