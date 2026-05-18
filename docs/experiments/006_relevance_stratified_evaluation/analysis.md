# Relevance-stratified evaluation

This analysis checks whether the land-use evidence extractor is useful as a
filter, even though the short `landuse_evidence_summary` text did not replace
raw article content as the best classifier input.

No LLM was rerun for this experiment. The analysis reuses the frozen Qwen and
Gemma prediction files from the original shuffled-control experiments, joins
them to `data/wiki/article_landuse_evidence_summaries.json`, and then recomputes
metrics by relevance, evidence type, and CORINE spatial-confidence subsets.

The generated artifacts are in:

```text
data/experiments/006_relevance_stratified_evaluation/article_text_classification_relevance_stratified_v1/
```

The main files are `overview_by_relevance.csv`,
`overview_by_relevance_and_spatial_confidence.csv`,
`overview_by_evidence_type.csv`, `shuffled_delta_by_relevance.csv`,
`majority_baselines_by_relevance.csv`, `per_class_corine_by_relevance.csv`,
`model_comparison_by_relevance.csv`, `evidence_metadata_distribution.csv`,
`manifest.json`, and `summary.md`.

## Validation

The manifest records `no_llm_rerun: true`. The evaluator loaded 18,312 frozen
prediction rows from:

- `data/experiments/001_qwen_e2e_shuffled_control/article_text_classification_e2e_with_shuffled_control_v1/`
- `data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/`

It joined those rows to:

- `data/wiki/article_landuse_evidence_summaries.json`
- `data/experiments/002_corine_spatial_confidence/corine_spatial_confidence_v1/spatial_confidence.csv`

The evidence metadata was available for all evaluated article ids. For the
article set, relevance was distributed as:

| Relevance | Articles |
|---|---:|
| none | 443 |
| low | 232 |
| medium | 280 |
| high | 296 |

So the key high-relevance group used below, `relevance_medium_high`, contains
576 articles.

## CORINE: relevance filtering strengthens the signal

For Qwen, filtering to medium/high evidence relevance improves every aligned
CORINE input:

| Text source | Balanced accuracy, all | Balanced accuracy, medium/high | Macro-F1, all | Macro-F1, medium/high |
|---|---:|---:|---:|---:|
| summary | 0.255 | 0.323 | 0.244 | 0.298 |
| summary_no_place | 0.243 | 0.306 | 0.235 | 0.277 |
| content | 0.281 | 0.374 | 0.270 | 0.346 |

Gemma shows the same CORINE pattern:

| Text source | Balanced accuracy, all | Balanced accuracy, medium/high | Macro-F1, all | Macro-F1, medium/high |
|---|---:|---:|---:|---:|
| summary | 0.236 | 0.312 | 0.217 | 0.286 |
| summary_no_place | 0.223 | 0.292 | 0.209 | 0.265 |
| content | 0.271 | 0.352 | 0.254 | 0.328 |

This is the strongest result of this analysis: the evidence extractor is useful
as a quality filter. When it says the article contains medium/high land-cover
evidence, the existing classifiers perform better, especially on raw content.

Raw content remains the best direct input. The relevance filter does not make
generic summaries beat content, but it makes all aligned text sources cleaner.

## Spatial confidence plus relevance is cleaner still

The cleanest CORINE signal comes from combining the relevance filter with the
existing spatial-confidence filter:

```text
point_label_share_250m >= 0.8 AND relevance in {medium, high}
```

For Qwen, raw content reaches balanced accuracy `0.409` and macro-F1 `0.363` on
321 CORINE examples. For Gemma, raw content reaches balanced accuracy `0.384`
and macro-F1 `0.355` on the same subset.

This supports the main scientific interpretation from the spatial-confidence
work: text is more predictive when the target label is spatially reliable. The
new evidence metadata adds a second useful filter: the article also needs to
contain explicit landscape or land-use information.

## Shuffled controls remain low

Inside the medium/high relevance subset, aligned text still beats shuffled text.
For CORINE, using balanced accuracy:

| Model | Text source | Aligned | Shuffled | Delta |
|---|---|---:|---:|---:|
| Qwen | summary | 0.323 | 0.059 | +0.265 |
| Qwen | summary_no_place | 0.306 | 0.055 | +0.250 |
| Qwen | content | 0.374 | 0.076 | +0.297 |
| Gemma | summary | 0.312 | 0.042 | +0.270 |
| Gemma | summary_no_place | 0.292 | 0.049 | +0.242 |
| Gemma | content | 0.352 | 0.074 | +0.279 |

This matters because relevance filtering could otherwise just select an easier
class distribution. The shuffled controls stay low, so the improved aligned
scores are still tied to article-specific text.

## CORINE per-class behavior

For raw content, medium/high relevance improves most CORINE classes.

Qwen content, all articles to medium/high:

| Class | Support all -> medium/high | F1 all -> medium/high |
|---|---:|---:|
| 21 arable land | 209 -> 51 | 0.292 -> 0.381 |
| 22 permanent crops | 137 -> 85 | 0.673 -> 0.873 |
| 23 pastures | 156 -> 52 | 0.117 -> 0.187 |
| 24 heterogeneous agriculture | 183 -> 79 | 0.208 -> 0.231 |
| 31 forests | 474 -> 253 | 0.381 -> 0.547 |
| 32 shrub/herbaceous vegetation | 50 -> 31 | 0.327 -> 0.455 |
| 51 inland waters | 42 -> 25 | 0.431 -> 0.442 |

Gemma content improves strongly for classes 21, 22, 23, 31, and 32, while class
24 and 51 are roughly flat or slightly lower:

| Class | Support all -> medium/high | F1 all -> medium/high |
|---|---:|---:|
| 21 arable land | 209 -> 51 | 0.160 -> 0.367 |
| 22 permanent crops | 137 -> 85 | 0.685 -> 0.883 |
| 23 pastures | 156 -> 52 | 0.058 -> 0.118 |
| 24 heterogeneous agriculture | 183 -> 79 | 0.257 -> 0.246 |
| 31 forests | 474 -> 253 | 0.581 -> 0.668 |
| 32 shrub/herbaceous vegetation | 50 -> 31 | 0.100 -> 0.218 |
| 51 inland waters | 42 -> 25 | 0.449 -> 0.447 |

Classes 33 and 41 have zero support in these runs, so they do not contribute
meaningful per-class evidence here.

## OSM: useful but model-dependent

OSM remains harder because it is a multi-label task and exact match is strict.
Still, relevance filtering helps several important rows.

For Qwen:

| Text source | Exact match, all | Exact match, medium/high | Micro-F1, all | Micro-F1, medium/high | Jaccard, all | Jaccard, medium/high |
|---|---:|---:|---:|---:|---:|---:|
| summary | 0.145 | 0.163 | 0.161 | 0.186 | 0.155 | 0.178 |
| summary_no_place | 0.164 | 0.203 | 0.189 | 0.236 | 0.180 | 0.226 |
| content | 0.164 | 0.196 | 0.197 | 0.236 | 0.190 | 0.236 |

For Gemma, the pattern is mixed:

| Text source | Exact match, all | Exact match, medium/high | Micro-F1, all | Micro-F1, medium/high | Jaccard, all | Jaccard, medium/high |
|---|---:|---:|---:|---:|---:|---:|
| summary | 0.240 | 0.242 | 0.264 | 0.276 | 0.261 | 0.269 |
| summary_no_place | 0.244 | 0.268 | 0.259 | 0.287 | 0.258 | 0.284 |
| content | 0.215 | 0.196 | 0.247 | 0.243 | 0.252 | 0.247 |

Gemma benefits most on `summary_no_place`, while its raw-content OSM exact match
does not improve under the medium/high relevance filter. On the combined
spatial + medium/high subset, Gemma `summary_no_place` is the best OSM row among
the inspected examples: exact match `0.294`, micro-F1 `0.323`, and Jaccard
`0.315` on 92 articles.

## Conclusion

The evidence extractor is useful, but not in the way first tested.

As a direct replacement text, `landuse_evidence_summary` was too lossy. As
metadata, however, it is valuable. The `landcover_relevance` score identifies
articles where existing raw content and summaries are more predictive, and this
effect is strongest when combined with high CORINE spatial confidence.

The best current interpretation is:

> The land-use evidence extractor is useful as a relevance and quality filter,
> even though its short summary is not rich enough to replace raw article
> content.

The next experiment should use this metadata to filter or weight the evaluation
set, and then test a less lossy evidence representation: for example a compact
evidence card with multiple factual bullets instead of a one-to-three-sentence
summary. Another useful next step is article/entity-type filtering with
Wikipedia categories or Wikidata, because many articles simply do not describe
land cover.
