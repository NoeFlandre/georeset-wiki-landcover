# Article-Text Classification E2E with Shuffled Control v1: Verified Analysis

This note analyzes the first frozen article-text classification batch before the CORINE spatial-confidence experiment.

Inputs analyzed:

- `data/experiments/article_text_classification_e2e_v1/`
- `data/experiments/article_text_classification_shuffled_control_v1/`
- `data/experiments/article_text_classification_e2e_with_shuffled_control_v1/`

The analysis is grounded in the generated `overview.csv`, `shuffled_delta.csv`, and per-run `*_metrics.json` files. No LLM predictions were rerun. The LLM used for all experiments here is Qwen3.6-27B-Q4_0.gguf

## 1. The Batch Is Complete and Parse-Clean

The combined frozen experiment contains 12 runs:

- 2 tasks: `corine_level2`, `osm`
- 6 text sources: `summary`, `summary_shuffled`, `summary_no_place`, `summary_no_place_shuffled`, `content`, `content_shuffled`

All evaluated runs have full coverage and zero parse errors:

| Task | Text source type | Eligible articles | Parse errors | Coverage |
| --- | --- | ---: | ---: | ---: |
| CORINE level-2 | each source | 1,251 | 0 | 1.000 |
| OSM | each source | 275 | 0 | 1.000 |

This means the first-batch interpretation is not confounded by incomplete parsing or retry failures.

## 2. CORINE Shows Real Signal Under Balanced Metrics

For CORINE, the primary balanced metric is macro recall. This is the balanced-accuracy view for single-label multiclass classification: each class contributes equally regardless of support.

| Text source | Accuracy | Macro recall | Macro-F1 | Majority macro recall | Delta vs majority macro recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| summary | 0.237 | 0.255 | 0.244 | 0.111 | +0.144 |
| summary no place | 0.232 | 0.243 | 0.235 | 0.111 | +0.132 |
| content | 0.293 | 0.281 | 0.270 | 0.111 | +0.170 |

All three aligned CORINE runs beat the balanced majority baseline. This is the first important positive result: Wikipedia text contains land-cover signal when evaluated with a metric that is not dominated by the majority class.

## 3. CORINE Raw Accuracy Is Misleading

The majority target share for CORINE is `0.379`. Raw model accuracy is lower than that for all aligned text sources:

| Text source | Model accuracy | Majority accuracy |
| --- | ---: | ---: |
| summary | 0.237 | 0.379 |
| summary no place | 0.232 | 0.379 |
| content | 0.293 | 0.379 |

If raw accuracy is treated as the main metric, the conclusion looks pessimistic. But the same runs beat majority strongly under macro recall. The correct interpretation is that the model is not simply learning the majority class; it recovers minority-class signal, but not enough to beat a majority-class baseline on raw accuracy.

## 4. Shuffled Controls Confirm the Signal Is Text-Linked

The shuffled controls keep targets and article eligibility but reassign texts across articles. For CORINE, aligned text is much stronger than shuffled text:

| Text source | Primary metric | Aligned score | Shuffled score | Delta | Aligned macro-F1 | Shuffled macro-F1 | Macro-F1 delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | macro recall | 0.255 | 0.063 | +0.191 | 0.244 | 0.070 | +0.174 |
| summary no place | macro recall | 0.243 | 0.058 | +0.184 | 0.235 | 0.068 | +0.167 |
| content | macro recall | 0.281 | 0.067 | +0.213 | 0.270 | 0.071 | +0.199 |

This is a strong diagnostic. The classifier is not only exploiting class priors; aligned article text carries information that shuffled text loses.

## 5. Raw Content Is the Best CORINE Text Source

Raw article content is the strongest CORINE source in this first batch:

| Text source | Accuracy | Macro recall | Macro-F1 |
| --- | ---: | ---: | ---: |
| summary | 0.237 | 0.255 | 0.244 |
| summary no place | 0.232 | 0.243 | 0.235 |
| content | 0.293 | 0.281 | 0.270 |

The generic summaries underperform raw content. This suggests that the summarization step likely removes useful land-cover cues. A better next text representation would be a landscape-evidence summary rather than a generic one-sentence summary.

## 6. CORINE Per-Class Behavior Is Uneven

For `corine_level2/content`, the model performs very differently by class:

| Class | Meaning | Support | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 21 | Arable land | 209 | 0.354 | 0.249 | 0.292 |
| 22 | Permanent crops | 137 | 0.892 | 0.540 | 0.673 |
| 23 | Pastures | 156 | 0.149 | 0.096 | 0.117 |
| 24 | Heterogeneous agriculture | 183 | 0.186 | 0.235 | 0.208 |
| 31 | Forests | 474 | 0.594 | 0.281 | 0.381 |
| 32 | Shrub/herbaceous | 50 | 0.315 | 0.340 | 0.327 |
| 51 | Inland waters | 42 | 0.297 | 0.786 | 0.431 |

Classes `33` and `41` have zero support in this evaluated article set and therefore have zero scores in the macro average.

The strongest class is `22` permanent crops, followed by useful but imperfect signal for forests and inland waters. The weakest classes are pastures and heterogeneous agricultural areas. This supports a class-aware conclusion: text is more useful for semantically salient or named landscape classes than for generic agricultural land-cover classes.

## 7. OSM Shows Signal Versus Shuffled, but Exact Match Remains Hard

For OSM, the strict primary metric is exact-match accuracy over the full multi-label set. All aligned OSM runs are below the most-frequent-label-set baseline of `0.207`. Because exact match is harsh for multi-label prediction, this section also reports mean per-article Jaccard similarity between the true and predicted label sets:

| Text source | Exact match | Majority label-set baseline | Delta vs majority | Jaccard | Majority Jaccard baseline | Jaccard delta vs majority | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.145 | 0.207 | -0.062 | 0.155 | 0.211 | -0.056 | 0.161 | 0.112 |
| summary no place | 0.164 | 0.207 | -0.044 | 0.180 | 0.211 | -0.031 | 0.188 | 0.122 |
| content | 0.164 | 0.207 | -0.044 | 0.190 | 0.211 | -0.021 | 0.197 | 0.158 |

The Jaccard majority baseline is computed by always predicting the most frequent OSM label set, `["grass"]`, then averaging per-article Jaccard against the true label set. The aligned runs still do not beat that baseline, but the gap is smaller under Jaccard than under exact match, especially for raw content.

However, aligned OSM text still beats shuffled text:

| Text source | Exact match | Shuffled exact match | Delta | Jaccard | Shuffled Jaccard | Jaccard delta | Macro-F1 | Shuffled macro-F1 | Macro-F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 0.145 | 0.076 | +0.069 | 0.155 | 0.082 | +0.073 | 0.112 | 0.039 | +0.073 |
| summary no place | 0.164 | 0.080 | +0.084 | 0.180 | 0.090 | +0.090 | 0.122 | 0.045 | +0.077 |
| content | 0.164 | 0.058 | +0.105 | 0.190 | 0.077 | +0.113 | 0.158 | 0.046 | +0.112 |

The OSM result is therefore mixed: exact-match and Jaccard performance are not yet above a simple label-set majority baseline, but aligned text has clear signal relative to shuffled text under exact match, Jaccard, and macro-F1. Exact match is also a harsh metric for a multi-label task, so Jaccard, micro-F1, macro-F1, and per-label behavior should be read alongside it.

## 8. OSM Per-Label Behavior Shows Salient Natural Features

For `osm/content`, the strongest supported labels include:

| Label | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| water | 21 | 0.463 | 0.905 | 0.613 |
| farmyard | 5 | 0.400 | 0.400 | 0.400 |
| vineyard | 4 | 0.200 | 0.500 | 0.286 |
| grassland | 43 | 0.333 | 0.209 | 0.257 |
| forest | 44 | 0.187 | 0.318 | 0.235 |

Several common labels remain weak, including `meadow` with F1 `0.039`, `scrub` with F1 `0.000`, and `orchard` with F1 `0.000`. This reinforces the same class-aware theme: labels that are more directly named or described in text are easier than generic or visually defined land-use labels.

## 9. Scientific Interpretation Before Spatial Confidence

The first batch already supported three claims:

1. Article text carries real land-cover signal, because aligned runs beat shuffled controls across both tasks.
2. Raw content is stronger than the current generic summaries, suggesting that summarization removes useful landscape evidence.
3. Majority baselines must be interpreted carefully. CORINE loses under raw accuracy but wins under balanced macro recall; OSM loses under strict exact-match majority but still beats shuffled controls.

The main unresolved issue in this first batch was ground-truth reliability. Because CORINE labels were assigned by point-in-polygon, it was unclear how much performance was limited by noisy spatial supervision. That uncertainty motivated the later spatial-confidence experiment.
