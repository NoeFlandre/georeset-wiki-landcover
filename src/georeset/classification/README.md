# Classification Package

This package implements the article-text land-cover classification experiment.
It supports CORINE level-2 single-label classification and OSM multi-label
classification under the same text-source and shuffled-control protocol.

## Files

- `__init__.py`: marks the package; keep it free of heavy imports.
- `ground_truth.py`: spatial joins that build task-specific ground truth from
  article coordinates and CORINE/OSM polygons.
- `labels.py`: CORINE level-2 descriptions and OSM label allowlists.
- `llm_classifier.py`: LLM boundary. It builds prompts, calls the shared Llama
  client, records model metadata, and converts inference/parser failures into
  auditable prediction records.
- `metrics.py`: single-label and multi-label metric implementations.
- `prediction_parser.py`: conservative parser/normalizer for model JSON or
  near-JSON outputs.
- `records.py`: prediction checkpoint shape and resumability skip policy.
- `runner.py`: reusable classification orchestration used by packaged CLIs and
  tests.
- `task_setup.py`: task-specific data loading and label-description setup.
- `text_sources.py`: primary and shuffled text-source policy.
- `types.py`: narrow typed contracts for classification results.

## Invariants

- CORINE stays single-label. Multiple valid CORINE labels are recorded as
  `parse_status="ambiguous"` and excluded from evaluated metrics.
- OSM stays multi-label. Predictions and ground truth may contain several valid
  labels.
- Seed, temperature, prompts, allowed labels, text source, model path, model
  repo ID, and `CLASSIFICATION_POLICY_VERSION` are part of the reproducibility
  contract.
- Shuffled controls change only text assignment, not targets or labels.

## Where To Add Changes

- Add new labels or label policies in `labels.py` and task setup tests.
- Add new text-source variants in `text_sources.py` and runner tests.
- Add model-call behavior at the `llm_classifier.py` boundary, not inside pure
  metrics or parser modules.
- Add experiment orchestration scripts under `georeset.cli`, not here, unless
  the code is reusable and testable.
