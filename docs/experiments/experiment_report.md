# GeoReset Experiment Report

This report consolidates the completed GeoReset experiments through experiment
013. It is written from the existing Markdown reports and generated artifacts in
`data/experiments/`; no experiments, LLM calls, GPU jobs, labels, or prediction
files were rerun for this synthesis.

The central research question across the project is:

> Can geolocated Wikipedia article text, article metadata, and weak spatial
> supervision produce useful land-cover signal for downstream geospatial
> learning?

The answer so far is nuanced but increasingly clear:

- Wikipedia text contains real land-cover signal.
- The signal is strongest for CORINE level-2 classification when article
  relevance and spatial label confidence are high.
- Raw content is still the strongest direct text source; deterministic evidence
  summaries, evidence cards, highlights, and retrieved windows are useful mainly
  for filtering, diagnostics, and cost reduction.
- OSM multi-label prediction is harder and more composition-sensitive. Some Qwen
  OSM subsets show real signal, but OSM claims must remain more cautious than
  CORINE claims.
- Strict weak-label filtering can hurt downstream image learning because it
  removes too many rare-class examples.

## Data, Tasks, And Metrics

The experiments use geolocated French Wikipedia articles joined to land-cover
labels and article-derived text sources.

Two prediction tasks recur throughout the experiments:

1. `corine_level2`: single-label CORINE level-2 classification.
2. `osm`: multi-label OSM land-use/land-cover label-set prediction.

For CORINE, the main metric is balanced accuracy, implemented in early reports
as `macro_recall`. It is the right primary metric because raw accuracy is highly
affected by class imbalance. A majority-class classifier can score high raw
accuracy on forest-heavy subsets while learning no minority-class signal.

For OSM, exact-match accuracy is very strict because the full predicted label
set must match the full target label set. Later reports therefore read exact
match together with Jaccard, micro-F1, and macro-F1. Jaccard is often the most
interpretable OSM metric because it gives partial credit for overlapping
multi-label sets.

The main text sources tested were:

- `summary`: generic place summary.
- `summary_no_place`: generic no-place summary with place-name cues removed.
- `content`: raw article content.
- `landuse_evidence_summary`: short extracted land-use evidence summary.
- `evidence_card`: deterministic structured evidence card.
- `content_with_evidence_card`: evidence card prepended to raw content.
- evidence-highlighted content and retrieved evidence windows.
- shuffled variants of each source, where targets are kept fixed but texts are
  reassigned across articles.

The shuffled controls are crucial. If aligned text beats shuffled text, the model
is using article-specific information rather than only class priors or prompt
artifacts.

## Experiment 001: Qwen E2E Article-Text Classification With Shuffled Controls

Report: `docs/experiments/001_qwen_e2e_shuffled_control/analysis.md`

Artifacts:

- `data/experiments/001_qwen_e2e_shuffled_control/article_text_classification_e2e_with_shuffled_control_v1/`

### What Was Done

The first frozen classification experiment used Qwen3.6-27B-Q4_0 to predict
CORINE and OSM labels from three aligned text sources and their shuffled
controls:

- `summary`
- `summary_no_place`
- `content`
- corresponding shuffled variants

The batch contained 12 runs: 2 tasks times 6 text sources. All runs were
parse-clean with full coverage:

- CORINE: 1,251 eligible articles per source.
- OSM: 275 eligible articles per source.
- Coverage: 1.0.
- Parse errors: 0.

### Results

For CORINE, raw content was the strongest source:

| source | accuracy | balanced accuracy | macro-F1 | shuffled balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| `summary` | 0.237 | 0.255 | 0.244 | 0.063 |
| `summary_no_place` | 0.232 | 0.243 | 0.235 | 0.058 |
| `content` | 0.293 | 0.281 | 0.270 | 0.067 |

Raw CORINE accuracy was below the majority-class raw accuracy baseline, but
balanced accuracy was well above the balanced majority baseline of 0.111. This
means Qwen was not merely predicting the dominant class; it recovered some
minority-class signal.

For OSM, all aligned runs were below the strict majority label-set exact-match
baseline of 0.207, but aligned text still beat shuffled text:

| source | exact match | shuffled exact match | micro-F1 | shuffled micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `summary` | 0.145 | 0.076 | 0.161 | 0.087 |
| `summary_no_place` | 0.164 | 0.080 | 0.188 | 0.098 |
| `content` | 0.164 | 0.058 | 0.197 | 0.097 |

### Interpretation

Experiment 001 established the first reliable signal:

- Article text is predictive for CORINE under balanced metrics.
- Raw content is stronger than generic summaries.
- Shuffled controls show that aligned text carries article-specific signal.
- OSM is harder: aligned text helps, but strict exact-match performance is still
  weak relative to the majority label-set baseline.

This experiment also exposed the next major uncertainty: point-in-polygon CORINE
labels might be spatially noisy.

## Experiment 002: CORINE Spatial Confidence

Report: `docs/experiments/002_corine_spatial_confidence/README.md`

Artifacts:

- `data/experiments/002_corine_spatial_confidence/corine_spatial_confidence_v1/`

### What Was Done

This analysis did not rerun any LLM. It measured how spatially reliable each
article's CORINE point label was by buffering article coordinates at 100 m,
250 m, 500 m, and 1000 m, then computing area-weighted CORINE level-2 label
shares inside each buffer.

The key variable is `point_label_share`: the share of surrounding area whose
dominant CORINE label matches the original point-in-polygon label. Computation
used EPSG:2154 and retained the full CORINE dataset, including artificial
classes, because nearby artificial land is real ambiguity evidence.

### Results

The confidence table covered all 1,251 CORINE-evaluable articles. Later
spatial-subset analyses showed the effective support at common thresholds:

| spatial condition | articles kept | share kept |
| --- | ---: | ---: |
| all available spatial confidence | 1,251 | 100.0% |
| dominant matches point label at 250 m | 1,082 | 86.5% |
| dominant matches point label at 500 m | 968 | 77.4% |
| point label share 250 m >= 0.8 | 646 | 51.6% |
| point label share 250 m >= 0.9 | 518 | 41.4% |
| point label share 500 m >= 0.8 | 419 | 33.5% |

### Interpretation

This experiment showed that point-in-polygon labels are often fragile. Only
about half of article points are clean at the 250 m / 0.8 threshold, and only
about one third are clean at the 500 m / 0.8 threshold. Spatial confidence is
therefore not just a diagnostic; it is necessary for interpreting text-label
agreement.

## Experiment 003: Qwen Spatial-Confidence Reevaluation

Report: `docs/experiments/003_qwen_spatial_confidence_reevaluation/analysis.md`

Artifacts:

- `data/experiments/003_qwen_spatial_confidence_reevaluation/article_text_classification_spatial_confidence_v1/`

### What Was Done

The frozen Qwen predictions from experiment 001 were reevaluated on spatially
reliable CORINE subsets. No LLM outputs were changed. The goal was to test
whether performance improves when the target label is more spatially credible.

### Results

For Qwen CORINE raw content:

| subset | n | accuracy | balanced accuracy | macro-F1 | shuffled balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| all spatial rows | 1,251 | 0.293 | 0.281 | 0.270 | 0.067 |
| 250 m share >= 0.8 | 646 | 0.350 | 0.352 | 0.285 | 0.071 |
| 250 m share >= 0.9 | 518 | 0.351 | 0.354 | 0.270 | 0.070 |
| 500 m share >= 0.8 | 419 | 0.334 | 0.302 | 0.213 | 0.067 |

The 250 m / 0.8 subset improved balanced accuracy from 0.281 to 0.352 while the
shuffled control stayed near 0.071. The improvement is therefore not explained
by a generic prompt or label prior.

Raw accuracy still stayed below the raw majority baseline because spatial
filtering made the subset more forest-dominated. For example, the majority raw
accuracy baseline rose from 0.379 on all rows to 0.502 on the 250 m / 0.8
subset and 0.659 on the 500 m / 0.8 subset.

For OSM raw content:

| subset | n | exact match | Jaccard | micro-F1 | shuffled exact match |
| --- | ---: | ---: | ---: | ---: | ---: |
| all spatial rows | 275 | 0.164 | 0.190 | 0.197 | 0.058 |
| 250 m share >= 0.8 | 160 | 0.225 | 0.245 | 0.249 | 0.063 |
| 500 m share >= 0.8 | 124 | 0.218 | 0.237 | 0.237 | 0.065 |

### Interpretation

Spatial confidence clarified the text signal. Qwen content becomes substantially
stronger on reliable CORINE labels, especially at the 250 m / 0.8 threshold. The
500 m / 0.8 subset is more conservative but more imbalanced and lower-support,
so it is better as a diagnostic than as the main subset.

## Experiment 004: Gemma 4 Rerun And Qwen-vs-Gemma Comparison

Report: `docs/experiments/004_gemma4_model_rerun_and_comparison/analysis.md`

Artifacts:

- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/`

### What Was Done

The full Qwen classification protocol was rerun with `gemma-4-31B-it-Q4_0.gguf`.
Tasks, prompts, labels, summaries, content, shuffled controls, seed, and
temperature were unchanged. The Gemma outputs were then compared to Qwen on both
the original run set and spatial-confidence subsets.

All 12 Gemma runs were parse-clean with coverage 1.0.

### Results

Gemma CORINE:

| source | accuracy | balanced accuracy | macro-F1 | shuffled balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| `summary` | 0.285 | 0.236 | 0.217 | 0.047 |
| `summary_no_place` | 0.237 | 0.223 | 0.209 | 0.049 |
| `content` | 0.357 | 0.271 | 0.254 | 0.066 |

Gemma OSM:

| source | exact match | micro-F1 | macro-F1 | shuffled exact match |
| --- | ---: | ---: | ---: | ---: |
| `summary` | 0.240 | 0.264 | 0.129 | 0.131 |
| `summary_no_place` | 0.244 | 0.259 | 0.153 | 0.127 |
| `content` | 0.215 | 0.247 | 0.158 | 0.087 |

Compared with Qwen:

- On CORINE raw content, Gemma had higher raw accuracy: 0.357 vs Qwen 0.293.
- Qwen had slightly higher balanced accuracy and macro-F1 on raw content:
  balanced accuracy 0.281 vs Gemma 0.271; macro-F1 0.270 vs Gemma 0.254.
- On OSM raw content, Gemma improved exact match: 0.215 vs Qwen 0.164, while
  both had similar macro-F1 around 0.158.
- On the CORINE 250 m / 0.8 spatial subset, Gemma raw accuracy rose to 0.455,
  but Qwen balanced accuracy remained higher: Qwen 0.352 vs Gemma 0.326.

### Interpretation

Gemma is more majority-class aligned: it improves raw accuracy, especially on
forest-heavy subsets, but Qwen is often stronger under balanced CORINE metrics.
Gemma is stronger on some strict OSM exact-match metrics, but that advantage
does not automatically imply better rare-label behavior.

The model comparison made the evaluation principle clear: raw accuracy alone is
not a sufficient metric for this project.

## Experiment 005: Land-Use Evidence Summaries

Report: `docs/experiments/005_landuse_evidence_summary/analysis.md`

Artifacts:

- `data/experiments/005_landuse_evidence_summary/article_text_classification_landuse_evidence_v1__qwen3_6_27b_q4_0/`
- `data/experiments/005_landuse_evidence_summary/article_text_classification_landuse_evidence_v1__gemma4_31b_it_q4_0/`
- `data/experiments/005_landuse_evidence_summary/landuse_evidence_comparison_v1/`

### What Was Done

This experiment created no-place land-use evidence summaries from article text
and then ran Qwen and Gemma classification using those summaries. The hypothesis
was that generic summaries were losing landscape evidence, and that a
land-use-focused summary might preserve the important cues while removing place
name leakage.

The classification text sources were:

- `landuse_evidence_summary`
- `landuse_evidence_summary_shuffled`

All classification runs were parse-clean.

### Results

Qwen land-use evidence summaries:

| task | source | n | primary score | macro-F1 | exact match | micro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CORINE | evidence summary | 1,251 | 0.184 | 0.203 |  |  |
| CORINE | shuffled | 1,251 | 0.047 | 0.058 |  |  |
| OSM | evidence summary | 275 | 0.109 | 0.120 | 0.109 | 0.195 |
| OSM | shuffled | 275 | 0.062 | 0.052 | 0.062 | 0.131 |

Gemma land-use evidence summaries:

| task | source | n | primary score | macro-F1 | exact match | micro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CORINE | evidence summary | 1,251 | 0.218 | 0.219 |  |  |
| CORINE | shuffled | 1,251 | 0.052 | 0.059 |  |  |
| OSM | evidence summary | 275 | 0.102 | 0.093 | 0.102 | 0.178 |
| OSM | shuffled | 275 | 0.087 | 0.043 | 0.087 | 0.127 |

Against raw content, land-use evidence summaries underperformed:

| model | task | evidence summary minus raw content, primary metric |
| --- | --- | ---: |
| Qwen | CORINE | -0.071 |
| Qwen | OSM | -0.036 |
| Gemma | CORINE | -0.018 |
| Gemma | OSM | -0.138 |

### Interpretation

The evidence summaries preserve article-specific signal because aligned evidence
summaries beat shuffled evidence summaries. However, they are not better direct
classifier inputs than raw content. The summaries are too compressed and lose
context that matters for classification.

Their main value is as metadata: they can identify relevant articles, evidence
density, uncertainty, and downstream quality signals.

## Experiment 006: Relevance-Stratified Evaluation

Report: `docs/experiments/006_relevance_stratified_evaluation/analysis.md`

Artifacts:

- `data/experiments/006_relevance_stratified_evaluation/article_text_classification_relevance_stratified_v1/`

### What Was Done

The project reused frozen Qwen and Gemma predictions and joined them to evidence
metadata from the land-use evidence extraction step. The analysis stratified
classification performance by:

- land-cover relevance: none, low, medium, high.
- evidence sentence count.
- uncertainty.
- spatial-confidence intersections.

No LLM outputs were changed.

### Results

For raw content, medium/high relevance strongly improved CORINE:

| model | task | subset | n | accuracy | balanced accuracy | macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | CORINE | all | 1,251 | 0.293 | 0.281 | 0.270 |
| Qwen | CORINE | medium/high relevance | 576 | 0.469 | 0.374 | 0.346 |
| Qwen | CORINE | high relevance | 296 | 0.564 | 0.390 | 0.356 |
| Gemma | CORINE | all | 1,251 | 0.357 | 0.271 | 0.254 |
| Gemma | CORINE | medium/high relevance | 576 | 0.526 | 0.352 | 0.328 |
| Gemma | CORINE | high relevance | 296 | 0.551 | 0.375 | 0.343 |

Combining relevance and spatial confidence was stronger:

| model | task | subset | n | accuracy | balanced accuracy | macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | CORINE | spatial 250 m >= 0.8 | 646 | 0.350 | 0.352 | 0.285 |
| Qwen | CORINE | medium/high relevance + spatial | 321 | 0.570 | 0.409 | 0.363 |
| Qwen | CORINE | high relevance + spatial | 157 | 0.707 | 0.436 | 0.397 |
| Gemma | CORINE | spatial 250 m >= 0.8 | 646 | 0.455 | 0.326 | 0.265 |
| Gemma | CORINE | medium/high relevance + spatial | 321 | 0.667 | 0.384 | 0.355 |
| Gemma | CORINE | high relevance + spatial | 157 | 0.713 | 0.428 | 0.397 |

For OSM, relevance helped but remained less stable:

| model | subset | n | exact match | micro-F1 |
| --- | --- | ---: | ---: | ---: |
| Qwen all | all | 275 | 0.164 | 0.197 |
| Qwen medium/high relevance | medium/high | 153 | 0.196 | 0.236 |
| Qwen medium/high relevance + spatial | spatial + relevance | 92 | 0.261 | 0.287 |
| Gemma all | all | 275 | 0.215 | 0.247 |
| Gemma medium/high relevance | medium/high | 153 | 0.196 | 0.243 |
| Gemma medium/high relevance + spatial | spatial + relevance | 92 | 0.261 | 0.279 |

### Interpretation

Relevance metadata became the most important non-spatial diagnostic. It shows
that the text classifier performs far better when the article actually contains
land-cover evidence. The best CORINE results come from the intersection of
medium/high relevance and 250 m spatial confidence.

For OSM, relevance and spatial filtering improve observed scores, but the
smaller support and harsher label-set evaluation require caution.

## Experiment 007: Article-Type And Relevance Stratification

Report: `docs/experiments/007_article_type_relevance_stratified_evaluation/analysis.md`

Artifacts:

- `data/experiments/007_article_type_relevance_stratified_evaluation/article_text_classification_article_type_relevance_stratified_v1/`

### What Was Done

This analysis added deterministic article-type assignments and asked whether
article type explains classifier behavior. Types included agriculture/vineyard,
natural landscape, water feature, settlement or administrative, built/cultural
site, transport infrastructure, person/event, and other/unclear.

The experiment then combined article type with relevance and spatial confidence.

### Results

Article type alone was informative but incomplete. A large share of articles
fell into `other_or_unclear`, mostly because category metadata was missing or
weak. Relevance still found useful signal inside that bucket.

Key raw-content rows:

| model | task | subset | n | accuracy / exact | balanced accuracy / Jaccard | macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | CORINE | other/unclear + medium/high relevance | 263 | 0.483 | 0.363 | 0.341 |
| Qwen | CORINE | other/unclear + medium/high relevance + spatial | 164 | 0.543 | 0.356 | 0.311 |
| Gemma | CORINE | other/unclear + medium/high relevance | 263 | 0.567 | 0.331 | 0.321 |
| Gemma | CORINE | other/unclear + medium/high relevance + spatial | 164 | 0.695 | 0.346 | 0.325 |
| Qwen | OSM | other/unclear + medium/high relevance | 68 | 0.206 | 0.260 | 0.145 |
| Qwen | OSM | other/unclear + medium/high relevance + spatial | 42 | 0.262 | 0.290 | 0.135 |
| Gemma | OSM | other/unclear + medium/high relevance | 68 | 0.206 | 0.259 | 0.127 |
| Gemma | OSM | other/unclear + medium/high relevance + spatial | 42 | 0.262 | 0.281 | 0.118 |

Small type buckets were often misleading. For example, the water-feature OSM
row had exact match and Jaccard around 0.833 on only 12 examples, but this was
heavily affected by label-set concentration and should not be treated as a
general model-skill result.

### Interpretation

Article type is useful context, but not enough as a standalone filter. The
evidence extractor remains more useful than article type because many pages have
missing or ambiguous category metadata. The strongest article-type insight is
that relevance still recovers meaningful examples inside the large
`other_or_unclear` bucket.

## Experiment 008: Supervision Quality Score

Report: `docs/experiments/008_supervision_quality_score/analysis.md`

Artifacts:

- `data/experiments/008_supervision_quality_score/article_text_supervision_quality_score_v1/`

### What Was Done

This analysis-only experiment combined the previous metadata signals into a
per-article supervision quality score. Inputs included:

- spatial confidence.
- land-cover relevance.
- evidence sentence count.
- uncertainty.
- article type.

It then assigned recommended-use buckets:

- `use_for_training`
- `use_for_evaluation_only`
- `inspect_manually`
- `exclude`

No predictions were rerun.

### Results

Quality-scored articles: 1,251.

Quality bins:

| bin | count |
| --- | ---: |
| low | 448 |
| medium | 141 |
| high | 295 |
| very high | 367 |

Recommended-use distribution:

| recommended use | count |
| --- | ---: |
| inspect manually | 479 |
| exclude | 451 |
| use for training | 195 |
| use for evaluation only | 126 |

For raw content, quality-plus-spatial was strong on CORINE:

| model | task | subset | n | accuracy / exact | balanced accuracy / Jaccard | macro-F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | CORINE | quality high/very high | 662 | 0.426 | 0.359 | 0.330 |
| Qwen | CORINE | quality high/very high + spatial | 388 | 0.490 | 0.394 | 0.332 |
| Qwen | CORINE | relevance medium/high + spatial | 321 | 0.570 | 0.409 | 0.363 |
| Gemma | CORINE | quality high/very high | 662 | 0.520 | 0.346 | 0.321 |
| Gemma | CORINE | quality high/very high + spatial | 388 | 0.629 | 0.375 | 0.334 |
| Gemma | CORINE | relevance medium/high + spatial | 321 | 0.667 | 0.384 | 0.355 |

For OSM, quality-plus-spatial improved observed scores but remained sensitive to
support and composition:

| model | subset | n | exact match | Jaccard | micro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen quality high/very high + spatial | 110 | 0.227 | 0.257 | 0.262 |
| Qwen relevance medium/high + spatial | 92 | 0.261 | 0.291 | 0.287 |
| Gemma quality high/very high + spatial | 110 | 0.236 | 0.269 | 0.264 |
| Gemma relevance medium/high + spatial | 92 | 0.261 | 0.294 | 0.279 |

### Interpretation

The quality score is a useful compact policy signal, but it mostly behaves like
a structured proxy for relevance plus spatial confidence. It is valuable for
dataset construction and triage, but the best CORINE classifier subset remained
the simpler relevance + spatial intersection.

The recommended-use buckets are conservative. They are appropriate for
downstream weak-supervision policies, but small buckets, especially
`use_for_evaluation_only` on OSM, should be treated carefully.

## Experiment 009: Evidence Cards

Report: `docs/experiments/009_evidence_card_text_source/analysis.md`

Artifacts:

- `data/experiments/009_evidence_card_text_source/article_text_evidence_card_v1__qwen3_6_27b_q4_0/`
- `data/experiments/009_evidence_card_text_source/evidence_card_comparison_v1/`

### What Was Done

This experiment tested deterministic evidence cards as a text source for Qwen.
Evidence cards are structured, compact representations of land-use evidence and
metadata. The experiment evaluated:

- `evidence_card`
- `evidence_card_shuffled`
- `content_with_evidence_card`
- `content_with_evidence_card_shuffled`

### Results

CORINE:

| source | balanced accuracy | macro-F1 | shuffled balanced accuracy |
| --- | ---: | ---: | ---: |
| raw `content` baseline | 0.281 | 0.270 | 0.067 |
| `content_with_evidence_card` | 0.285 | 0.260 | 0.089 |
| `evidence_card` | 0.224 | 0.226 | 0.069 |

OSM:

| source | exact match | Jaccard | micro-F1 | shuffled exact match |
| --- | ---: | ---: | ---: | ---: |
| raw `content` baseline | 0.164 | 0.190 | 0.197 | 0.058 |
| `content_with_evidence_card` | 0.135 | 0.164 | 0.152 | 0.047 |
| `evidence_card` | 0.095 | 0.137 | 0.168 | 0.058 |

### Interpretation

Evidence cards preserve signal relative to shuffled controls, but they do not
improve over raw content as a direct classifier input. Prepending a card to raw
content slightly improved CORINE balanced accuracy over raw content, but reduced
macro-F1 and hurt OSM.

The evidence card is better understood as an interpretable diagnostic and
metadata artifact than as the main text source.

## Experiment 010: Evidence-Highlighted Content

Report: `docs/experiments/010_evidence_highlighted_content/analysis.md`

Artifacts:

- `data/experiments/010_evidence_highlighted_content/article_text_evidence_highlights_v1__qwen3_6_27b_q4_0/`
- `data/experiments/010_evidence_highlighted_content/article_text_evidence_highlights_v1__gemma4_31b_it_q4_0/`
- `data/experiments/010_evidence_highlighted_content/evidence_highlights_comparison_v1/`

### What Was Done

This experiment marked evidence-bearing spans inside raw content and classified
the highlighted content with Qwen and Gemma. The goal was to preserve raw
article context while drawing model attention to relevant evidence.

No new LLM summarization was used to create the highlighted representation.

### Results

All new Qwen and Gemma highlighted-content classification outputs were
parse-clean:

- Qwen CORINE aligned/shuffled: 1,251/1,251 ok.
- Gemma CORINE aligned/shuffled: 1,251/1,251 ok.
- Qwen OSM aligned/shuffled: 275/275 ok.
- Gemma OSM aligned/shuffled: 275/275 ok.

Aligned highlighted content beat shuffled highlighted content for both CORINE
and OSM, so the representation preserved article-specific signal:

| model | task | primary metric | aligned | shuffled | delta |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen | CORINE | balanced accuracy | 0.273 | 0.090 | +0.183 |
| Gemma | CORINE | balanced accuracy | 0.259 | 0.075 | +0.184 |
| Qwen | OSM | Jaccard | 0.148 | 0.065 | +0.084 |
| Gemma | OSM | Jaccard | 0.207 | 0.091 | +0.116 |

However, highlighted content did not consistently beat raw content. It improved
over short land-use evidence summaries for CORINE, but raw content remained the
strongest direct classifier input overall, especially when OSM was included.

All-article CORINE comparison:

| model | source | accuracy | balanced accuracy | macro-F1 |
| --- | --- | ---: | ---: | ---: |
| Qwen | raw content | 0.293 | 0.281 | 0.270 |
| Qwen | highlighted content | 0.313 | 0.273 | 0.256 |
| Gemma | raw content | 0.357 | 0.271 | 0.254 |
| Gemma | highlighted content | 0.321 | 0.259 | 0.253 |

All-article OSM comparison:

| model | source | exact match | Jaccard | micro-F1 | macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | raw content | 0.164 | 0.190 | 0.197 | 0.158 |
| Qwen | highlighted content | 0.102 | 0.148 | 0.180 | 0.116 |
| Gemma | raw content | 0.215 | 0.252 | 0.247 | 0.158 |
| Gemma | highlighted content | 0.153 | 0.207 | 0.218 | 0.150 |

On the combined `relevance_medium_high_and_spatial_250m_ge_0.8` subset,
highlighted content improved relative to all rows but still usually trailed raw
content:

| model | task | raw content | highlighted content |
| --- | --- | ---: | ---: |
| Qwen | CORINE balanced accuracy | 0.409 | 0.361 |
| Gemma | CORINE balanced accuracy | 0.384 | 0.379 |
| Qwen | OSM Jaccard | 0.291 | 0.234 |
| Gemma | OSM Jaccard | 0.294 | 0.223 |

### Interpretation

Highlighting is useful for diagnostics and possibly prompt-cost reduction, but
it did not justify replacing raw content. This reinforced a pattern that had
already appeared in experiments 005 and 009: deterministic evidence artifacts
are useful for filtering and interpretation, not necessarily as direct
classifier inputs.

## Experiment 011: Retrieved Evidence Windows

Report: `docs/experiments/011_retrieved_evidence_windows/analysis.md`

Artifacts:

- `data/experiments/011_retrieved_evidence_windows/retrieved_evidence_windows_comparison_v1/`

### What Was Done

This experiment retrieved local article windows around land-cover evidence
sentences. It compared:

- full raw content.
- retrieved evidence windows.
- retrieved evidence windows with place-name variants removed.
- evidence sentences only.
- random sentence windows.
- shuffled controls.

The goal was to test whether a compact, targeted raw-text representation could
replace full content.

The generated retrieval artifact contained 1,251 article records. Evidence text
was matched in 752 articles. Among those matched articles, the mean was 4.948
matched evidence sentences and 10.297 retrieved sentences after adding local
context. Articles without matched evidence fell back to the first sentence so
coverage remained complete and comparable.

The classifier protocol stayed fixed:

- models: Qwen3.6-27B-Q4_0 and gemma-4-31B-it-Q4_0.
- tasks: CORINE and OSM.
- seed: 42.
- temperature: 0.0.
- CORINE examples: 1,251 per run.
- OSM examples: 275 per run.
- all 20 jobs completed with zero parse errors.

### Results

For CORINE, retrieved evidence windows were stronger than random windows and
sentence-only retrieval, but still generally below full content.

Representative all-row CORINE balanced accuracy:

| model | source | balanced accuracy | macro-F1 |
| --- | --- | ---: | ---: |
| Qwen | full content | 0.281 | 0.270 |
| Qwen | retrieved evidence windows | 0.266 | 0.257 |
| Gemma | full content | 0.271 | 0.254 |
| Gemma | retrieved evidence windows | 0.245 | 0.234 |
| Gemma | retrieved windows no-place | 0.251 | 0.238 |

Shuffled controls stayed low. For example, retrieved-window CORINE deltas over
shuffled controls were:

| model | task | subset | aligned score | shuffled score | delta |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen | CORINE | all | 0.266 | 0.087 | +0.179 |
| Qwen | CORINE | medium/high relevance | 0.335 | 0.068 | +0.267 |
| Gemma | CORINE | all | 0.245 | 0.077 | +0.168 |
| Gemma | CORINE | medium/high relevance | 0.326 | 0.073 | +0.254 |

For OSM, retrieved windows did not solve the task. The experiment report
concludes that OSM labels are more local, sparse, and multi-label, while the
retrieval heuristic is based on broader land-cover evidence sentences.

### Interpretation

Retrieved windows are the best compact text representation tested so far for
CORINE, but they still do not replace full raw content. Full articles contain
useful indirect cues outside the retrieved evidence sentences.

The no-place result is also important: removing place-name variants did not
collapse performance, and for Gemma it slightly improved CORINE balanced
accuracy. The signal is therefore not only memorized place-name association.

## Experiment 012: CLIP Linear Probe With Weak Labels

Report: `docs/experiments/012_clip_linear_probe_weak_labels/analysis.md`

Artifacts:

- `data/experiments/012_clip_linear_probe_weak_labels/clip_linear_probe_weak_labels_v1/`

### What Was Done

This experiment moved from text classification to downstream image learning. It
used article coordinates to fetch Sentinel-2 RGB patches, embedded them with the
frozen image encoder from `openai/clip-vit-base-patch32`, and trained a NumPy
softmax linear probe for CORINE level-2 prediction.

It also evaluated out-of-the-box zero-shot CLIP using class-description prompts.

Setup:

- Sentinel-2 L2A RGB from Microsoft Planetary Computer.
- Date range: 2022-04-01 to 2022-10-31.
- Cloud filter: <25.
- Patch size: 224 x 224.
- Frozen CLIP image embeddings: 512 dimensions.
- Strict evaluation split: 35 examples, 5 per class.

Training tiers:

- `all`: all available weak labels after excluding evaluation pageids.
- `spatial_only`: 250 m spatial-confidence filter.
- `quality_spatial`: spatial plus medium/high relevance, high/very-high quality,
  and no high uncertainty.
- `text_spatial_agreement`: quality-spatial plus Qwen and Gemma both agreeing
  with the CORINE point label.

### Results

| tier | train examples | eval examples | accuracy | balanced accuracy | macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 482 | 35 | 0.600 | 0.600 | 0.586 |
| spatial_only | 357 | 35 | 0.571 | 0.571 | 0.589 |
| quality_spatial | 184 | 35 | 0.514 | 0.514 | 0.490 |
| text_spatial_agreement | 142 | 35 | 0.400 | 0.400 | 0.363 |
| zero-shot CLIP | 0 | 35 | 0.200 | 0.200 | 0.224 |

The best linear probe improved over zero-shot CLIP by +0.400 balanced accuracy
and about +0.362 macro-F1.

### Interpretation

The trained linear probe clearly beats out-of-the-box zero-shot CLIP. However,
the strictest weak-label filters hurt downstream performance. They remove noisy
examples, but also remove too many rare-class examples. The lesson is that
quality filters should not be used only as hard exclusions for downstream image
training at this data size.

The better policy is to use broad training sets with quality signals as weights,
sampling controls, evaluation strata, or calibration signals. A Sentinel-native
image encoder is also a natural follow-up.

## Experiment 013: Subset Randomization Controls

Report: `docs/experiments/013_subset_randomization_controls/analysis.md`

Artifacts:

- `data/experiments/013_subset_randomization_controls/article_text_subset_randomization_controls_v1/`

### What Was Done

This analysis-only experiment tested whether the filtered-subset gains reported
earlier survive randomization controls. It did not rerun LLMs, prompts, labels,
spatial confidence, or GPU jobs.

For each observed subset, model, task, and text source, the experiment compared
the observed score to two Monte Carlo controls with 1,000 draws:

- `random_same_n`: random subsets with the same number of examples.
- `random_same_target_distribution`: random subsets with the same CORINE target
  counts or OSM target-label-set counts.

The comparison universe was explicit: same parent experiment, model, task, text
source, and subset-specific metadata availability. Rows with `n < 30` were
marked `unstable_small_n=true`.

### Results

Headline raw-content rows:

| model | task | subset | n | observed | target-matched mean | conclusion |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen | CORINE | relevance medium/high | 576 | 0.374 | 0.280 | beats target-matched |
| Qwen | CORINE | spatial 250 m >= 0.8 | 646 | 0.352 | 0.280 | beats target-matched |
| Qwen | CORINE | relevance + spatial | 321 | 0.409 | 0.281 | beats target-matched |
| Qwen | CORINE | quality + spatial | 388 | 0.394 | 0.281 | beats target-matched |
| Gemma | CORINE | relevance medium/high | 576 | 0.352 | 0.271 | beats target-matched |
| Gemma | CORINE | spatial 250 m >= 0.8 | 646 | 0.326 | 0.271 | beats target-matched |
| Gemma | CORINE | relevance + spatial | 321 | 0.384 | 0.272 | beats target-matched |
| Gemma | CORINE | quality + spatial | 388 | 0.375 | 0.272 | beats target-matched |
| Qwen | OSM | relevance medium/high | 153 | 0.236 | 0.215 | beats same-n only |
| Qwen | OSM | spatial 250 m >= 0.8 | 160 | 0.245 | 0.206 | beats target-matched |
| Qwen | OSM | relevance + spatial | 92 | 0.291 | 0.231 | beats target-matched |
| Qwen | OSM | quality + spatial | 110 | 0.257 | 0.218 | beats same-n only |
| Gemma | OSM | relevance medium/high | 153 | 0.247 | 0.272 | below random |
| Gemma | OSM | relevance + spatial | 92 | 0.294 | 0.295 | not distinguishable |

For CORINE aligned-vs-shuffled deltas, the random controls were also strong.
For raw content, Qwen and Gemma CORINE relevance/spatial headline subsets had
observed deltas above the random 97.5% interval.

For OSM, delta controls were weaker. Qwen spatial-only OSM content survived, but
many OSM rows did not beat target-matched controls. Gemma OSM headline rows were
mostly not distinguishable after target matching.

### Interpretation

Experiment 013 is the strongest reviewer-proofing result so far.

The strongest CORINE claim survives: geolocated Wikipedia text is most
predictive when article relevance and spatial label reliability are credible,
and that effect is not explained away by support size or target class
composition.

The OSM claim must be softer: OSM text signal exists for some Qwen subsets, but
target composition and small support explain more of the apparent gains than
they do for CORINE.

## Experiment 014: Quality-Weighted Multiscale Image Probe

Report:
`docs/experiments/014_quality_weighted_multiscale_image_probe/analysis.md`

Artifacts:

- `data/experiments/014_quality_weighted_multiscale_image_probe/quality_weighted_multiscale_image_probe_v1/`

### What Was Done

This experiment implements the next-stage image-probe pipeline motivated by
Experiment 012. The scientific goal is to avoid the rare-class loss caused by
hard filtering weak labels. Instead of only dropping examples, it builds
quality, relevance, spatial-confidence, and Qwen/Gemma agreement signals into
soft sample weights.

The completed run was the staged MVP, not the full encoder/window grid. It used:

- encoder: frozen `openai/clip-vit-base-patch32`;
- Sentinel-2 RGB windows: 320 m and 2240 m;
- output image size: 224 x 224 pixels;
- embeddings: 512 dimensions;
- evaluation: the same 35-example strict split style, 5 examples per CORINE
  class;
- training policies: broad unweighted, spatial-only, quality-spatial hard
  filter, text-spatial-agreement hard filter, quality-weighted, class-balanced
  quality-weighted, spatial soft weighting, and text-agreement soft weighting;
- metrics: accuracy, allowed-label balanced accuracy, supported-label balanced
  accuracy, allowed/supported macro-F1, per-class metrics, bootstrap intervals,
  and confusion matrices.

The run did not rerun LLMs or change labels. It reused existing quality scores,
spatial metadata, and frozen Qwen/Gemma content predictions.

### Results

The MVP completed end-to-end. It produced 1,251 cached Sentinel-2 patches for
each physical window and 1,251 CLIP embeddings for each window. The probe output
contains 1,664 metric rows, 127,104 prediction rows, per-class metrics,
bootstrap confidence intervals, confusion matrices, and run manifests.

The model-performance result is negative:

| run | window | eval examples | accuracy | supported balanced accuracy | supported macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Experiment 014 trained classifier | 320 m | 35 | 0.143 | 0.143 | 0.036 |
| Experiment 014 trained classifier | 2240 m | 35 | 0.143 | 0.143 | 0.036 |
| Experiment 012 zero-shot CLIP baseline | prior single-scale patch | 35 | 0.200 | 0.200 | 0.224 |
| Experiment 012 best linear probe | prior single-scale patch | 35 | 0.600 | 0.600 | 0.586 |

The Experiment 014 MVP classifier often collapsed to predicting dominant CORINE
class `31`. Both tested physical scales therefore underperformed the earlier
zero-shot CLIP baseline and were far below the earlier broad weak-label linear
probe.

### Interpretation

The engineering result is positive: the new multiscale/weighted pipeline runs
end-to-end on Grid5000 and produces auditable artifacts. The scientific result
for this MVP configuration is negative: `clip_base` on 320 m and 2240 m crops
does not recover useful seven-class CORINE signal under the new split and
training setup.

This should not be overclaimed as a failure of all multiscale or quality-weighted
image learning. It says that the first MVP configuration is not enough. The next
decision should come from the planned full encoder/window run, especially the
stronger CLIP and DINOv2 encoders, before spending compute on random training
controls.

## Cross-Experiment Findings

### 1. Raw Content Remains The Best Direct Text Source

Generic summaries, land-use summaries, evidence cards, highlighted content, and
retrieved windows all preserve some article-specific signal, but none
consistently replaced raw content as the best direct classifier input.

The likely reason is that full articles contain indirect but useful cues:
article type, surrounding prose, named features, historical land use, nearby
landscape descriptions, and contextual terms that deterministic compression
removes.

### 2. Spatial Confidence Is Essential

Point-in-polygon labels are noisy. The 250 m / 0.8 `point_label_share` threshold
keeps about half the data and gives the best balance between reliability and
support. The 500 m / 0.8 threshold is stricter but too forest-dominated for
general evaluation.

### 3. Relevance Is The Strongest Metadata Filter

Medium/high land-cover relevance strongly improves CORINE, and the intersection
with spatial confidence gives the best observed balanced accuracy. Experiment
013 confirms that this is not merely a sample-size or target-composition
artifact for CORINE.

### 4. Quality Scores Are Useful But Should Be Used Carefully

Quality scores combine spatial, relevance, evidence, uncertainty, and article
type signals into a convenient triage policy. They are useful for data
management and downstream splits. But for CORINE, quality-plus-spatial behaves
mostly like a proxy for relevance-plus-spatial and does not beat the simplest
best filter.

For image training, strict quality filters reduce rare-class support too much.

### 5. Qwen And Gemma Have Different Biases

Qwen is stronger on balanced CORINE behavior. Gemma often has higher raw
accuracy and stronger OSM exact match, but this can reflect majority-class or
label-set alignment. Both models have real aligned-vs-shuffled signal, but model
comparisons must use balanced and subset-aware metrics.

### 6. OSM Is Harder Than CORINE

OSM exact match is a difficult multi-label target. The label sets are sparse,
local, and composition-sensitive. Some Qwen OSM subsets survive controls, but
Gemma OSM content often does not survive target matching. The OSM evidence is
real but weaker and should be framed as diagnostic rather than solved.

### 7. Downstream Image Learning Needs More Than The First MVP

The original CLIP linear probe showed that broad weak labels can train an image
model better than out-of-the-box zero-shot CLIP, while strict text/spatial
agreement filters underperform because they remove too much data. Experiment 014
validated the new multiscale/weighted image-probe pipeline, but its first
`clip_base` MVP at 320 m and 2240 m was chance-like. The image direction remains
open, but the next positive claim must come from the full encoder/window grid,
not from the MVP.

## Current Best Claims

The most defensible claims after all experiments are:

1. Geolocated Wikipedia text contains real land-cover signal.
2. The signal is strongest and most rigorously supported for CORINE level-2
   classification.
3. For CORINE, relevance and 250 m spatial confidence select examples where
   text prediction improves beyond random same-size and target-matched controls.
4. Raw content should remain the primary direct text source.
5. Evidence summaries, evidence cards, highlights, and retrieved windows should
   be used for filtering, diagnostics, interpretability, and cost reduction, not
   as replacements for raw content.
6. OSM has useful signal but is more sensitive to target composition and small
   support.
7. For downstream Sentinel patch learning, the original broad weak-label CLIP
   probe beat strict agreement-only training, but the newer multiscale MVP did
   not yet recover useful signal.

## Claims That Should Be Avoided Or Rephrased

The experiments do not support saying:

- The classifier is generally accurate under raw accuracy.
- OSM prediction is solved.
- Evidence summaries or cards are better than full article content.
- Strict text/spatial/model-agreement filtering is the best image-training
  policy.
- High scores on small article-type or recommended-use subsets are definitive.

Safer phrasing:

- CORINE text signal is robust under balanced metrics and strongest on relevant,
  spatially reliable articles.
- OSM shows article-specific signal, especially for Qwen in selected subsets,
  but target composition and support size explain more of the observed gains.
- Quality and evidence artifacts are best treated as selection and diagnostic
  signals.

## Recommended Next Experiments

The next high-value research steps are:

1. Train a downstream image model using broad weak labels with quality-aware
   weighting rather than hard filtering.
2. Compare CLIP with a Sentinel-native encoder, since generic CLIP zero-shot
   alignment is weak on CORINE satellite patches.
3. Keep the strict high-quality set as evaluation or calibration data, not as
   the only training set.
4. Run the full Experiment 014 encoder/window grid before interpreting the
   multiscale weighting idea.
5. Continue reporting CORINE with balanced accuracy and macro-F1, and OSM with
   Jaccard plus exact match.
6. For every future filtered subset, include same-size and target-matched random
   controls by default.

## Artifact Index

Primary reports:

- `docs/experiments/001_qwen_e2e_shuffled_control/analysis.md`
- `docs/experiments/002_corine_spatial_confidence/README.md`
- `docs/experiments/003_qwen_spatial_confidence_reevaluation/analysis.md`
- `docs/experiments/004_gemma4_model_rerun_and_comparison/analysis.md`
- `docs/experiments/005_landuse_evidence_summary/analysis.md`
- `docs/experiments/006_relevance_stratified_evaluation/analysis.md`
- `docs/experiments/007_article_type_relevance_stratified_evaluation/analysis.md`
- `docs/experiments/008_supervision_quality_score/analysis.md`
- `docs/experiments/009_evidence_card_text_source/analysis.md`
- `docs/experiments/010_evidence_highlighted_content/analysis.md`
- `docs/experiments/011_retrieved_evidence_windows/analysis.md`
- `docs/experiments/012_clip_linear_probe_weak_labels/analysis.md`
- `docs/experiments/013_subset_randomization_controls/analysis.md`
- `docs/experiments/014_quality_weighted_multiscale_image_probe/analysis.md`

Primary generated artifact roots:

- `data/experiments/001_qwen_e2e_shuffled_control/`
- `data/experiments/002_corine_spatial_confidence/`
- `data/experiments/003_qwen_spatial_confidence_reevaluation/`
- `data/experiments/004_gemma4_model_rerun_and_comparison/`
- `data/experiments/005_landuse_evidence_summary/`
- `data/experiments/006_relevance_stratified_evaluation/`
- `data/experiments/007_article_type_relevance_stratified_evaluation/`
- `data/experiments/008_supervision_quality_score/`
- `data/experiments/009_evidence_card_text_source/`
- `data/experiments/010_evidence_highlighted_content/`
- `data/experiments/011_retrieved_evidence_windows/`
- `data/experiments/012_clip_linear_probe_weak_labels/`
- `data/experiments/013_subset_randomization_controls/`
- `data/experiments/014_quality_weighted_multiscale_image_probe/`
