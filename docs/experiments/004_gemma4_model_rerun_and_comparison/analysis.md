# Gemma 4 31B IT Q4_0 Classification Rerun Analysis

This report analyzes the Gemma rerun of the article-text land-cover
classification experiment and the Qwen-vs-Gemma comparison. Every quantitative
claim below was checked against the generated CSV/JSON artifacts on
2026-05-10.

## Source Artifacts

Gemma frozen experiment:

- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/overview.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/shuffled_delta.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/manifest.json`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/summary.md`
- 12 `*_predictions.json` files and 12 `*_metrics.json` files

Gemma spatial reevaluation:

- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/overview_spatial_subsets.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/subset_counts.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/shuffled_delta_spatial_subsets.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/majority_baselines_spatial_subsets.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/per_class_metrics_corine_spatial_subsets.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/class_distribution_by_spatial_subset.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/manifest.json`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/summary.md`

Model comparison:

- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/qwen_vs_gemma_overview.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/qwen_vs_gemma_spatial_subsets.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/qwen_vs_gemma_shuffled_delta.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/qwen_vs_gemma_shuffled_delta_spatial_subsets.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/qwen_vs_gemma_per_class_corine.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/qwen_vs_gemma_prediction_distribution.csv`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/manifest.json`
- `data/experiments/004_gemma4_model_rerun_and_comparison/model_comparison_qwen_vs_gemma4_31b_it_q4_0/summary.md`

## Verification Status

The Gemma non-spatial overview has 12 rows: 2 tasks x 6 text sources.

The Gemma spatial-subset overview has 72 rows: 2 tasks x 6 text sources x 6
spatial subsets.

The comparison tables have:

- 12 rows in `qwen_vs_gemma_overview.csv`
- 72 rows in `qwen_vs_gemma_spatial_subsets.csv`
- 6 rows in `qwen_vs_gemma_shuffled_delta.csv`
- 324 rows in `qwen_vs_gemma_per_class_corine.csv`

All 12 Gemma prediction files and all 12 Gemma metrics files are present.
Across the Gemma overview, total `n_parse_error` is 0 and every run has
`coverage = 1.0`. Each Gemma prediction file contains only
`parse_status == "ok"` records.

The Gemma run used:

- model: `gemma-4-31B-it-Q4_0.gguf`
- model repo: `unsloth/gemma-4-31B-it-GGUF`
- seed: 42
- temperature: 0.0

The experiment changed the classifier model only. The tasks, labels, prompts,
summaries, raw content, shuffled-control logic, and spatial-confidence table
were reused.

## Gemma Non-Spatial Results

| task | text source | n | primary metric | primary score | accuracy | exact match | macro recall | macro F1 | micro F1 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CORINE | summary | 1251 | macro recall | 0.2358 | 0.2846 |  | 0.2358 | 0.2172 |  |
| CORINE | summary_no_place | 1251 | macro recall | 0.2232 | 0.2374 |  | 0.2232 | 0.2086 |  |
| CORINE | content | 1251 | macro recall | 0.2714 | 0.3565 |  | 0.2714 | 0.2544 |  |
| OSM | summary | 275 | exact match | 0.2400 |  | 0.2400 | 0.1811 | 0.1287 | 0.2640 |
| OSM | summary_no_place | 275 | exact match | 0.2436 |  | 0.2436 | 0.1971 | 0.1529 | 0.2585 |
| OSM | content | 275 | exact match | 0.2145 |  | 0.2145 | 0.3221 | 0.1584 | 0.2472 |

For CORINE, Gemma performs best on raw `content` under macro recall and raw
accuracy. `content` reaches accuracy 0.3565, macro recall 0.2714, and macro F1
0.2544.

For OSM, Gemma's strict exact-match score is highest on `summary_no_place`
at 0.2436, followed closely by `summary` at 0.2400. Raw `content` has lower
exact match at 0.2145, but it has the highest macro recall at 0.3221. This
suggests raw content recovers more true labels, but not always the exact full
label set.

## Gemma Shuffled-Control Results

| task | text source | primary delta aligned - shuffled | macro F1 delta | micro F1 delta |
| --- | --- | ---: | ---: | ---: |
| CORINE | summary | 0.1892 | 0.1657 |  |
| CORINE | summary_no_place | 0.1745 | 0.1528 |  |
| CORINE | content | 0.2057 | 0.1874 |  |
| OSM | summary | 0.1091 | 0.0655 | 0.1051 |
| OSM | summary_no_place | 0.1164 | 0.0939 | 0.1070 |
| OSM | content | 0.1273 | 0.0999 | 0.1133 |

All Gemma aligned text sources beat their matching shuffled controls on the
task primary metric. The largest CORINE primary delta is for raw content
at +0.2057 macro recall. The largest OSM primary delta is also for raw content
at +0.1273 exact-match accuracy.

This supports the conclusion that the Gemma predictions use article-specific
text signal rather than only target priors or label imbalance.

## Gemma Spatial-Confidence Results

For `content`, the spatial subsets show the following:

| task | subset | n | accuracy | balanced accuracy | exact match | micro F1 | macro F1 | Jaccard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CORINE | all_available_spatial_confidence | 1251 | 0.3565 | 0.2714 |  |  | 0.2544 |  |
| CORINE | point_label_share_250m_ge_0.8 | 646 | 0.4551 | 0.3259 |  |  | 0.2646 |  |
| CORINE | point_label_share_250m_ge_0.9 | 518 | 0.4846 | 0.3481 |  |  | 0.2640 |  |
| CORINE | point_label_share_500m_ge_0.8 | 419 | 0.4964 | 0.3168 |  |  | 0.2257 |  |
| OSM | all_available_spatial_confidence | 275 |  |  | 0.2145 | 0.2472 | 0.1584 | 0.2521 |
| OSM | point_label_share_250m_ge_0.8 | 160 |  |  | 0.2562 | 0.2786 | 0.1778 | 0.2865 |
| OSM | point_label_share_250m_ge_0.9 | 140 |  |  | 0.2500 | 0.2698 | 0.1497 | 0.2748 |
| OSM | point_label_share_500m_ge_0.8 | 124 |  |  | 0.2581 | 0.2801 | 0.1166 | 0.2871 |

For CORINE content, the high-purity 250 m subset improves Gemma from accuracy
0.3565 to 0.4551 and from balanced accuracy 0.2714 to 0.3259. Macro F1 rises
more modestly, from 0.2544 to 0.2646.

For OSM content, the high-purity 250 m subset improves Gemma from exact match
0.2145 to 0.2562, micro F1 from 0.2472 to 0.2786, macro F1 from 0.1584 to
0.1778, and Jaccard from 0.2521 to 0.2865.

The spatial-confidence pattern is therefore positive for Gemma: higher CORINE
spatial reliability is associated with better content-based classification for
both CORINE and OSM. The effect is stronger for accuracy/exact-match than for
macro F1, which means part of the improvement may come from easier or more
imbalanced high-purity subsets.

## Qwen vs Gemma: Main Comparison

For the primary `content` runs:

| task | metric | Qwen | Gemma | Gemma - Qwen |
| --- | --- | ---: | ---: | ---: |
| CORINE content | accuracy | 0.2934 | 0.3565 | +0.0631 |
| CORINE content | macro recall / balanced accuracy | 0.2807 | 0.2714 | -0.0093 |
| CORINE content | macro F1 | 0.2699 | 0.2544 | -0.0155 |
| OSM content | exact match | 0.1636 | 0.2145 | +0.0509 |
| OSM content | micro F1 | 0.1966 | 0.2472 | +0.0506 |
| OSM content | macro F1 | 0.1584 | 0.1584 | approximately 0.0000 |

For the main high-confidence subset `point_label_share_250m_ge_0.8`:

| task | metric | Qwen | Gemma | Gemma - Qwen |
| --- | --- | ---: | ---: | ---: |
| CORINE content | accuracy | 0.3498 | 0.4551 | +0.1053 |
| CORINE content | balanced accuracy | 0.3518 | 0.3259 | -0.0259 |
| CORINE content | macro F1 | 0.2850 | 0.2646 | -0.0204 |
| OSM content | exact match | 0.2250 | 0.2562 | +0.0312 |
| OSM content | micro F1 | 0.2486 | 0.2786 | +0.0300 |
| OSM content | macro F1 | 0.2039 | 0.1778 | -0.0261 |
| OSM content | Jaccard | 0.2453 | 0.2865 | +0.0412 |

Gemma is stronger than Qwen on raw CORINE accuracy and OSM exact-match/micro
metrics. Qwen remains stronger on CORINE balanced accuracy and macro F1,
especially in the high-confidence subset. This means Gemma is not uniformly
better: it appears more aligned with common classes and exact-match outcomes,
while Qwen distributes performance more evenly across CORINE classes.

## CORINE Per-Class Behavior

For CORINE content on the `point_label_share_250m_ge_0.8` subset:

| CORINE label | support | Qwen F1 | Gemma F1 | Gemma - Qwen |
| --- | ---: | ---: | ---: | ---: |
| 21 | 100 | 0.3097 | 0.1250 | -0.1847 |
| 22 | 81 | 0.8112 | 0.8028 | -0.0084 |
| 23 | 50 | 0.0440 | 0.0364 | -0.0076 |
| 24 | 65 | 0.2249 | 0.2710 | +0.0462 |
| 31 | 324 | 0.4508 | 0.6813 | +0.2306 |
| 32 | 14 | 0.3913 | 0.1569 | -0.2344 |
| 33 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 41 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 51 | 12 | 0.3333 | 0.3077 | -0.0256 |

Gemma's biggest gain over Qwen is class `31` forests, where F1 rises from
0.4508 to 0.6813. Gemma also improves class `24` heterogeneous agricultural
areas. Qwen is substantially stronger on class `21` arable land and class `32`
shrub/herbaceous vegetation. Class `23` pastures remains weak for both models.

This per-class table explains why Gemma can improve raw accuracy while losing
balanced metrics: the high-confidence subset is forest-heavy, and Gemma is much
stronger on forests but weaker on several minority classes.

## Prediction Distribution Check

For CORINE content, the top predicted labels differ:

| model | top predictions |
| --- | --- |
| Qwen | `33`: 290 (0.2318), `24`: 231 (0.1847), `31`: 224 (0.1791), `21`: 147 (0.1175), `51`: 111 (0.0887) |
| Gemma | `31`: 363 (0.2902), `24`: 307 (0.2454), `33`: 243 (0.1942), `51`: 105 (0.0839), `22`: 82 (0.0655) |

Gemma predicts forest (`31`) much more often than Qwen. This is consistent with
the per-class result: Gemma's forest F1 improves substantially, but the stronger
forest tendency also helps explain weaker balanced performance on some minority
classes.

## Interpretation

The Qwen findings are partially robust to Gemma. Both model families show
positive aligned-minus-shuffled deltas, so both extract article-specific
land-cover signal. Both also improve on spatially reliable subsets, supporting
the broader claim that text signal becomes clearer when point-based CORINE
ground truth is spatially reliable.

The model comparison also refines the conclusion. Gemma is better for raw
accuracy and OSM exact/micro metrics in the tested content runs, but Qwen is
better for balanced CORINE behavior. Therefore the research result should not
be framed as one model simply dominating the other. The stronger conclusion is:

> Article text contains real land-cover signal across two local LLM families,
> but the measured benefit depends strongly on metric choice, class imbalance,
> spatial ground-truth reliability, and class semantics.

For CORINE, balanced accuracy, macro F1, per-class metrics, and shuffled
controls remain more informative than raw accuracy alone. For OSM, exact match
is useful but harsh; micro F1, macro F1, and Jaccard should be interpreted
alongside exact match.

## Recommended Next Step

The next high-impact step is not another rerun at a different temperature.
Temperature stayed fixed at 0.0 and should remain fixed for protocol control.

The next useful experiment is to improve the evidence text itself, for example
with a `landscape_evidence_summary` that preserves land-cover cues instead of
generic article summaries. The current model comparison suggests that raw
content often carries more useful signal than generic summaries, especially for
CORINE content.
