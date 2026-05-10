# Spatial Package

This package contains spatial-policy and spatial-confidence utilities.

## Modules

- `policy.py`: shared point/polygon predicate policy used to keep filtering and
  classification ground-truth behavior aligned.
- `corine_confidence.py`: computes CORINE level-2 area shares around article
  coordinates at fixed metric radii.

## CORINE Confidence Contract

- Reprojects article points and CORINE polygons to EPSG:2154 before buffering
  or area computation.
- Uses the full CORINE dataset, including artificial classes, because nearby
  artificial land is ambiguity evidence.
- Computes area-weighted label shares through spatial-index candidate filtering.
- Keeps `point_label_share` as the primary confidence variable because it
  measures whether the original point-in-polygon target dominates the buffer.

## Experiment Relationship

Spatial confidence is computed once and reused to reevaluate frozen prediction
sets. It must not rerun the LLM or mutate parent experiment outputs.
