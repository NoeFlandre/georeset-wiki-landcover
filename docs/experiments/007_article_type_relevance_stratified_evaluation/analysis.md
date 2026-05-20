# Article-Type + Relevance Stratified Evaluation v1

This experiment asks whether French Wikipedia category metadata helps explain where GeoReset Wiki Land-Cover article-text classification works.

It is an analysis-only experiment. No LLM was rerun. The analysis reused the frozen Qwen and Gemma prediction files, the existing land-use evidence metadata, and the existing CORINE spatial-confidence table. The new signal is a deterministic, category-derived article-type proxy built from French Wikipedia categories and page metadata.

The output artifacts are in:

```text
data/experiments/007_article_type_relevance_stratified_evaluation/article_text_classification_article_type_relevance_stratified_v1/
```

## What Was Added

For each article, the new metadata fetcher retrieved French Wikipedia categories and page properties without refetching article content. A deterministic rule set then assigned:

- one `primary_article_type`;
- all `candidate_article_types`;
- `matched_categories`;
- `matched_rules`;
- category availability counts.

The taxonomy is intentionally simple:

```text
water_feature
natural_landscape
agriculture_or_vineyard
built_or_cultural_site
transport_infrastructure
settlement_or_administrative
person_or_event
other_or_unclear
```

This is not a gold annotation. It is a noisy metadata proxy. The report should be read in that spirit.

## Metadata Coverage

The fetch produced article-type metadata for all 1,251 article pageids used in the classification experiments. However, only 605 articles had visible category metadata returned by the API. The remaining 646 articles therefore fall into `other_or_unclear`.

| Primary article type | Articles |
|---|---:|
| other_or_unclear | 672 |
| settlement_or_administrative | 284 |
| natural_landscape | 115 |
| built_or_cultural_site | 94 |
| water_feature | 36 |
| agriculture_or_vineyard | 34 |
| transport_infrastructure | 14 |
| person_or_event | 2 |

This matters. Article type is useful, but the metadata is incomplete and uneven. The largest bucket is not a semantic class; it is mostly missing or unmatched category metadata.

The relevance metadata is more directly tied to land-cover evidence. Medium/high relevance is common in the expected article types:

| Primary article type | Medium/high relevance share |
|---|---:|
| agriculture_or_vineyard | 97.1% |
| water_feature | 91.7% |
| natural_landscape | 80.9% |
| built_or_cultural_site | 55.3% |
| other_or_unclear | 39.1% |
| settlement_or_administrative | 34.9% |
| transport_infrastructure | 7.1% |

So the category proxy and the evidence extractor agree in broad strokes: landscape, water, and agriculture/vineyard pages are often relevant. But the category proxy alone is not enough to guarantee better classification performance.

## CORINE Findings

For CORINE, raw content remains the most important text source to inspect. The key metric is balanced accuracy / macro recall, with macro-F1 as the second main metric. Raw accuracy is secondary because the class distribution is strongly imbalanced.

### Article Type Alone

For Qwen with raw content:

| Subset | n | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| other_or_unclear | 672 | 0.269 | 0.257 | 0.255 |
| agriculture_or_vineyard | 34 | 0.912 | 0.222 | 0.183 |
| natural_landscape | 115 | 0.426 | 0.204 | 0.203 |
| settlement_or_administrative | 284 | 0.243 | 0.119 | 0.123 |
| built_or_cultural_site | 94 | 0.234 | 0.112 | 0.140 |
| water_feature | 36 | 0.306 | 0.111 | 0.052 |

The high raw accuracy for `agriculture_or_vineyard` is mostly a class-imbalance effect. Its majority baseline accuracy is already 0.882. Balanced accuracy and macro-F1 are much more modest.

For Gemma with raw content, the same pattern holds. Gemma is stronger on raw accuracy in several buckets, but Qwen often remains competitive or stronger on balanced behavior.

### Article Type + Relevance

Adding `landcover_relevance` is more informative than article type alone. For Qwen raw content on medium/high relevance articles:

| Subset | n | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| other_or_unclear + medium/high relevance | 263 | 0.483 | 0.363 | 0.341 |
| natural_landscape + medium/high relevance | 93 | 0.505 | 0.229 | 0.217 |
| settlement_or_administrative + medium/high relevance | 99 | 0.354 | 0.225 | 0.200 |
| agriculture_or_vineyard + medium/high relevance | 33 | 0.939 | 0.222 | 0.183 |
| built_or_cultural_site + medium/high relevance | 52 | 0.365 | 0.182 | 0.190 |

The strongest balanced CORINE result in this table is still the large `other_or_unclear` bucket when filtered by relevance. That is important: category metadata is incomplete, but the evidence extractor can still identify articles with useful land-cover signal even when category metadata is missing or unclear.

The aligned-vs-shuffled deltas remain positive inside medium/high relevance article-type subsets. For Qwen raw content, balanced-accuracy deltas against shuffled content are:

| Subset | Aligned | Shuffled | Delta |
|---|---:|---:|---:|
| other_or_unclear + medium/high relevance | 0.363 | 0.073 | +0.290 |
| agriculture_or_vineyard + medium/high relevance | 0.222 | 0.011 | +0.211 |
| natural_landscape + medium/high relevance | 0.229 | 0.059 | +0.169 |
| settlement_or_administrative + medium/high relevance | 0.225 | 0.074 | +0.151 |

This confirms that the signal is still article-specific, not just a category or label-frequency artifact.

### Article Type + Relevance + Spatial Confidence

The cleanest CORINE diagnostic still combines text relevance with spatial reliability. Within `point_label_share_250m >= 0.8`, Qwen raw content reaches:

| Subset | n | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| other_or_unclear + medium/high relevance + spatial 250m | 164 | 0.543 | 0.356 | 0.311 |
| settlement_or_administrative + medium/high relevance + spatial 250m | 31 | 0.516 | 0.336 | 0.291 |
| natural_landscape + medium/high relevance + spatial 250m | 43 | 0.744 | 0.275 | 0.264 |
| agriculture_or_vineyard + medium/high relevance + spatial 250m | 29 | 0.966 | 0.222 | 0.220 |

The spatial filter improves the credibility of the supervision, but some groups become small and class-dominated. Those rows should be treated as diagnostics, not as final benchmark rankings.

## OSM Findings

OSM remains harder because it is multi-label and exact match is strict.

For Qwen raw content by article type:

| Subset | n | Exact match | Micro-F1 | Macro-F1 | Jaccard |
|---|---:|---:|---:|---:|---:|
| water_feature | 12 | 0.833 | 0.833 | 0.061 | 0.833 |
| other_or_unclear | 143 | 0.168 | 0.212 | 0.150 | 0.197 |
| settlement_or_administrative | 34 | 0.118 | 0.118 | 0.057 | 0.132 |
| built_or_cultural_site | 44 | 0.114 | 0.119 | 0.068 | 0.125 |
| natural_landscape | 37 | 0.054 | 0.168 | 0.106 | 0.108 |

The `water_feature` row looks high, but its most-frequent-label-set baseline is also 0.833, so this is not strong evidence of classifier skill by itself. The more useful signal is that medium/high relevance improves the larger `other_or_unclear` bucket: Qwen OSM raw-content Jaccard rises to 0.260 for `other_or_unclear + medium/high relevance`, compared with 0.197 for all `other_or_unclear` articles.

Gemma shows a similar pattern and is stronger than Qwen on several OSM content rows, especially `other_or_unclear` and `built_or_cultural_site`, but the same caution applies: small article-type buckets and harsh exact-match scoring make OSM conclusions less stable than CORINE conclusions.

## Main Conclusion

The article-type proxy is useful as an explanatory diagnostic, but it is not a better filter than the land-use evidence relevance score.

The strongest finding is:

```text
Evidence relevance remains the main filter. Article type helps interpret where the relevance signal comes from, but category-derived type alone is too incomplete and noisy to replace relevance filtering.
```

This supports the previous relevance-stratified conclusion. The land-use evidence extractor is useful less as a replacement text source and more as a relevance/quality signal. Article categories add context: agriculture/vineyard, water, and natural-landscape pages are usually relevant, while settlement, transport, and many unclear pages are mixed. But because many articles have missing or weak category metadata, the evidence extractor still finds useful signal inside the large `other_or_unclear` bucket.

## Recommended Next Experiment

The next experiment should not simply add more category rules. The useful direction is to combine the relevance score with a richer article-level representation:

```text
Use raw content or a compact evidence-card representation, but weight or filter examples by landcover_relevance and spatial confidence.
```

A good next scoped step would be an evaluation that uses relevance/spatial confidence as sample weights or reporting strata, rather than a hard filter only. That would preserve more data while still accounting for noisy supervision and irrelevant articles.
