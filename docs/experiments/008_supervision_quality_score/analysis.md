# Supervision Quality Score v1

This is an analysis-only experiment that re-scores the frozen article-text
classification predictions from Qwen and Gemma before computing final subset metrics.
No LLM rerun is performed.

The experiment reads these inputs:

- `data/wiki/wiki_articles.json`
- `data/wiki/article_landuse_evidence_summaries.json`
- `data/experiments/corine_spatial_confidence_v1/spatial_confidence.csv`
- `data/experiments/article_text_classification_article_type_relevance_stratified_v1/article_type_assignments.csv`
- `data/experiments/article_text_classification_e2e_with_shuffled_control_v1/*`
- `data/experiments/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/*`

Outputs are written to:

```text
data/experiments/article_text_supervision_quality_score_v1/
```

The experiment id used in the generated manifest is
`article_text_supervision_quality_score_v1`.

## Scoring method

Each article receives a deterministic quality score from a fixed rule set:

- `relevance_score`: `none=0, low=1, medium=2, high=3, missing/unknown=0`
- `spatial_score`: `3 * point_label_share_250m` (or `0` if missing)
- `evidence_density_score`: `0` for missing/`0`, `1` for `1`, `2` for `>=2`
- `article_type_score`: `2` for `agriculture_or_vineyard`, `natural_landscape`,
  `water_feature`; `1` for `settlement_or_administrative`, `built_or_cultural_site`;
  otherwise `0`
- `uncertainty_penalty`: `low=0`, `medium=1`, `high=2`, `missing/unknown=1`

Then:

```text
quality_score = relevance_score + spatial_score + evidence_density_score + article_type_score - uncertainty_penalty
```

Bins:

- `quality_low`: `< 3`
- `quality_medium`: `3 <= score < 5`
- `quality_high`: `5 <= score < 7`
- `quality_very_high`: `>= 7`

`recommended_use` is assigned by conservative precedence:
`exclude > use_for_evaluation_only > use_for_training > inspect_manually`.
In particular, `exclude` overrides all other rules when quality is low,
`landcover_relevance == none`, or `uncertainty == high`.

## What changed from previous experiments

This pass computes one deterministic row per `pageid` and carries:

- `pageid` string key and coordinates
- evidence summary stats
- spatial confidence shares (250 m and 500 m)
- candidate/primary article-type metadata
- computed quality components and final score/bin
- `recommended_use`

These rows are written both to `quality_scores.*` and
`candidate_training_pairs_by_quality.*`.

## Generated quality distribution

From the generated files:

- `quality_low`: 448
- `quality_medium`: 141
- `quality_high`: 295
- `quality_very_high`: 367
- total quality-scored articles: 1251

Recommended-use distribution:

- `inspect_manually`: 479
- `exclude`: 451
- `use_for_training`: 195
- `use_for_evaluation_only`: 126

## Main metric findings

The new outputs include recomputed metrics for all required subsets and both
text sources/models/tasks. A few grounded findings:

### CORINE (`corine_level2`)

- For `content/all`, balanced accuracy:
  - Qwen: `0.280709`
  - Gemma: `0.271401`
- For `content/recommended_use_evaluation_only` (`n=126`), balanced accuracy:
  - Qwen: `0.472684`
  - Gemma: `0.471955`
- For `content/quality_high_or_very_high` (`n=662`), balanced accuracy:
  - Qwen: `0.358592`
  - Gemma: `0.345856`
- For `content/quality_high_or_very_high_and_spatial_250m_ge_0.8` (`n=388`),
  balanced accuracy:
  - Qwen: `0.393544`
  - Gemma: `0.375037`

Shuffled-control deltas stay positive on the same subsets; selected examples:

- `content/quality_high_or_very_high_and_spatial_250m_ge_0.8`:
  - Qwen delta `+0.3120` (`balanced_accuracy`)
  - Gemma delta `+0.3126` (`balanced_accuracy`)

### OSM (`osm`)

- For `content/all`, exact-match accuracy:
  - Qwen: `0.163636`
  - Gemma: `0.214545`
- For `content/quality_high_or_very_high` (`n=174`), exact-match accuracy:
  - Qwen: `0.183908`
  - Gemma: `0.201149`
- For `content/recommended_use_evaluation_only` (`n=25`), exact-match accuracy:
  - Qwen: `0.36`
  - Gemma: `0.36`
  and Jaccard:
  - Qwen: `0.406667`
  - Gemma: `0.415333`

Shuffled deltas on strict OSM metrics remain large:

- `content/recommended_use_evaluation_only`: exact-match deltas are around
  `+0.24` for Gemma and `+0.28` for Qwen on this subset.

## Simple-filter comparison

The `overview_comparison_simple_filters.csv` comparison shows that the new
quality filters are not a clear replacement for the simpler `relevance+spatial`
heuristic:

- CORINE/content (`content`, balanced accuracy):
  - Qwen:
    - `quality_high_or_very_high = 0.358591808` vs
      relevance `0.373542640`, spatial `0.351753569`,
      relevance+spatial `0.409148614`, article_type `0.322556751`.
      -> better only than spatial and article_type; worse than relevance and
      relevance+spatial.
    - `quality_high_or_very_high_and_spatial_250m_ge_0.8 = 0.393543830` vs
      relevance `0.373542640`, spatial `0.351753569`,
      relevance+spatial `0.409148614`, article_type `0.322556751`.
      -> better than relevance, spatial, article_type; worse than relevance+spatial.
  - Gemma:
    - `quality_high_or_very_high = 0.345856168` vs
      relevance `0.352125510`, spatial `0.325886583`,
      relevance+spatial `0.384421673`, article_type `0.321048916`.
      -> better only than spatial and article_type; worse than relevance and
      relevance+spatial.
    - `quality_high_or_very_high_and_spatial_250m_ge_0.8 = 0.375037477` vs
      relevance `0.352125510`, spatial `0.325886583`,
      relevance+spatial `0.384421673`, article_type `0.321048916`.
      -> better than relevance, spatial, article_type; slightly worse than
      relevance+spatial.
- OSM/content (`exact_match_accuracy` and `jaccard`) tells a similar story:
  - Qwen:
    - `quality_high_or_very_high` exact/jaccard: `0.222222222` / `0.226327944`.
      Relevance: `0.236383442` / `0.235897436`; spatial: `0.2453125` / `0.248618785`;
      relevance+spatial: `0.260869565` / `0.286995516`; article_type:
      `0.274509804` / `0.281481481`.
      -> worse than relevance, spatial, relevance+spatial, and article_type.
    - `quality_high_or_very_high_and_spatial_250m_ge_0.8` exact/jaccard:
      `0.256818182` / `0.261538462`. It is above relevance
      (`0.236383442` / `0.235897436`) and spatial (`0.2453125` / `0.248618785`)
      on exact match, but still below relevance+spatial
      (`0.260869565` / `0.286995516`) and article_type (`0.274509804` / `0.281481481`)
      on both metrics.
  - Gemma:
    - `quality_high_or_very_high` exact/jaccard: `0.248563218` / `0.245833333`.
      Relevance: `0.246732026` / `0.242562929`; spatial: `0.286458333` / `0.278606965`;
      relevance+spatial: `0.294384058` / `0.278884462`; article_type:
      `0.345098039` / `0.328767123`.
      -> slightly above relevance-only, but below spatial, relevance+spatial, and
      article_type on both metrics.
    - `quality_high_or_very_high_and_spatial_250m_ge_0.8` exact/jaccard:
      `0.268939394` / `0.263888889` is above relevance (`0.246732026` / `0.242562929`)
      but below spatial (`0.286458333` / `0.278606965`), relevance+spatial
      (`0.294384058` / `0.278884462`), and article_type (`0.345098039` / `0.328767123`)
      on exact and jaccard.
- Article-type-only baseline remains useful for interpretation but does not dominate
  the spatial+relevance filters:
  - CORINE/content `article_type_high_prior`: Qwen `0.322556751`, Gemma
    `0.321048916` (both lower than relevance+spatial; Qwen/Gemma quality+spatial are
    above article-type only).
  - OSM/content `article_type_high_prior`: Qwen exact `0.274509804` / jaccard
    `0.281481481`; Gemma exact `0.345098039` / jaccard `0.328767123` (both are
    above the corresponding OSM quality rows, so article type adds interpretation but
    does not dominate the combined relevance/spatial signal).

Conclusion: `quality_high_or_very_high_and_spatial_250m_ge_0.8` is not
consistently stronger than the simpler relevance-only, spatial-only, or
relevance+spatial candidates. It appears useful as a conservative candidate-pruning
signal, but not as a clear dominant replacement for the simpler filters.

## What to use next

The new quality score is best used to generate training/evaluation candidates, not
as a standalone replacement signal. The strict exclusion rule is intentionally
conservative to keep noisy or weak supervision from flowing into downstream data
selection.

The strongest diagnostic rows are:

- `use_for_evaluation_only` and `use_for_training` buckets under strong spatial
  confidence (`>= 0.8` at 250 m),
- `quality_high_or_very_high` and especially the same subset intersected with
  spatial confidence.

Because all output files are generated from frozen artifacts, rerunning this CLI is
fast and repeatable for future label-model combinations.
