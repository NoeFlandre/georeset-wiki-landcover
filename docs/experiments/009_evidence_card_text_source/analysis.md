# Evidence-card text-source experiment

This experiment tested one question: can we turn the metadata we already built
into a deterministic "evidence card" that helps the classifier more than the
previous short land-use evidence summary?

No LLM was used to create the cards. The cards were built from existing
metadata: land-cover relevance, evidence sentences, article type, CORINE spatial
confidence, and the supervision-quality score. The classifier was rerun only for
Qwen on four new text sources:

- `evidence_card`
- `evidence_card_shuffled`
- `content_with_evidence_card`
- `content_with_evidence_card_shuffled`

The `evidence_card` source is compact and no-place. The
`content_with_evidence_card` source is not no-place, because it keeps the
original raw Wikipedia article after the card. This variant tests whether a
structured card can guide the classifier while preserving the raw article
signal.

Source artifacts:

- Frozen run:
  `data/experiments/article_text_evidence_card_v1__qwen3_6_27b_q4_0/`
- Comparison tables:
  `data/experiments/evidence_card_comparison_v1/`
- Cards:
  `data/wiki/article_evidence_cards.json`

All eight evidence-card runs are parse-clean: CORINE has 1,251/1,251 valid
predictions, and OSM has 275/275 valid predictions. During the run we found one
important card-design issue: raw internal enum tokens such as `exclude` and
`settlement_or_administrative` confused the OSM classifier. The card text was
changed to use human-readable French labels while keeping the raw enum fields in
JSON for inspection.

## Main non-spatial results

For CORINE on all articles:

| text source | accuracy | balanced accuracy | macro-F1 |
|---|---:|---:|---:|
| content | 0.293 | 0.281 | 0.270 |
| content_with_evidence_card | 0.337 | 0.285 | 0.260 |
| summary | 0.237 | 0.255 | 0.244 |
| summary_no_place | 0.232 | 0.243 | 0.235 |
| evidence_card | 0.286 | 0.224 | 0.226 |
| landuse_evidence_summary | 0.237 | 0.184 | 0.203 |

The evidence card is clearly better than the previous
`landuse_evidence_summary` as a compact representation. It improves CORINE
balanced accuracy from 0.184 to 0.224 and macro-F1 from 0.203 to 0.226.

However, the card alone does not beat raw content or the generic summaries on
balanced CORINE metrics. It is a better compact diagnostic text than the first
evidence summary, but it is not a replacement for raw content.

`content_with_evidence_card` gives the highest CORINE accuracy, 0.337, compared
with 0.293 for raw content. But balanced accuracy only moves from 0.281 to
0.285, and macro-F1 drops from 0.270 to 0.260. So the card helps the model choose
more majority-like correct labels, but it does not clearly improve the balanced
class behavior we care about most.

For OSM on all articles:

| text source | exact match | micro-F1 | macro-F1 | Jaccard |
|---|---:|---:|---:|---:|
| content | 0.164 | 0.197 | 0.158 | 0.190 |
| summary_no_place | 0.164 | 0.188 | 0.122 | 0.180 |
| landuse_evidence_summary | 0.109 | 0.195 | 0.120 | 0.168 |
| content_with_evidence_card | 0.135 | 0.152 | 0.086 | 0.164 |
| evidence_card | 0.095 | 0.168 | 0.107 | 0.137 |

For OSM, raw content remains the best direct input. The evidence card improves
over its shuffled control, but it does not beat raw content, no-place summary,
or the previous land-use evidence summary on the main OSM metrics.

## Shuffled controls

The new text sources are still article-linked. On CORINE all articles:

- `evidence_card` balanced accuracy: 0.224
- `evidence_card_shuffled`: 0.069
- delta: +0.155

For `content_with_evidence_card`:

- aligned balanced accuracy: 0.285
- shuffled balanced accuracy: 0.089
- delta: +0.195

On OSM all articles:

- `evidence_card` Jaccard: 0.137
- `evidence_card_shuffled`: 0.085
- delta: +0.052
- `content_with_evidence_card` Jaccard: 0.164
- shuffled version: 0.075
- delta: +0.089

So the new sources are not just reproducing label priors. They carry
article-specific signal, but the signal is not strong enough to outperform raw
content.

## Quality and spatial subsets

The evidence-card result is stronger on cleaner subsets, but the ordering stays
the same: raw content remains the best balanced CORINE input.

On `relevance_medium_high_and_spatial_250m_ge_0.8` for CORINE:

| text source | n | accuracy | balanced accuracy | macro-F1 |
|---|---:|---:|---:|---:|
| content | 321 | 0.570 | 0.409 | 0.363 |
| content_with_evidence_card | 321 | 0.604 | 0.362 | 0.323 |
| evidence_card | 321 | 0.601 | 0.354 | 0.316 |
| landuse_evidence_summary | 321 | 0.592 | 0.360 | 0.315 |

The card variants have very high raw accuracy on this clean subset, but raw
content still has the best balanced accuracy and macro-F1. This suggests the
card emphasizes strong/easy classes, especially forests, rather than improving
minority-class balance.

For OSM on the same subset:

| text source | n | exact match | micro-F1 | macro-F1 | Jaccard |
|---|---:|---:|---:|---:|---:|
| content | 92 | 0.261 | 0.287 | 0.181 | 0.291 |
| content_with_evidence_card | 92 | 0.239 | 0.255 | 0.133 | 0.270 |
| evidence_card | 92 | 0.163 | 0.247 | 0.169 | 0.230 |
| landuse_evidence_summary | 92 | 0.120 | 0.253 | 0.156 | 0.219 |

Again, the card helps compared with the previous short summary in some compact
settings, but raw content remains the strongest overall.

## Per-class CORINE behavior

On all articles, the most useful class-level gains from the card are:

- class 21, arable land: content F1 0.292, evidence_card F1 0.333,
  content_with_evidence_card F1 0.369
- class 31, forests: content F1 0.381, evidence_card F1 0.436,
  content_with_evidence_card F1 0.440
- class 22, permanent crops: all strong, around 0.67 F1

But the card hurts or does not solve several difficult classes:

- class 23, pastures: still weak, around 0.11 F1
- class 24, heterogeneous agriculture: evidence_card drops to 0.100 F1
  compared with 0.208 for raw content
- class 32, shrub/herbaceous vegetation: evidence_card drops to 0.111 F1
  compared with 0.327 for raw content
- class 51, inland waters: evidence_card drops to 0.277 F1 compared with
  0.431 for raw content

This explains why raw accuracy can improve while macro-F1 does not. The card
helps some common or semantically clear classes, but it loses information needed
for minority or more nuanced landscape classes.

## Conclusion

The evidence card is a better compact representation than the previous
`landuse_evidence_summary`, but raw content remains the strongest classifier
input.

The most accurate interpretation is:

> The card organizes useful metadata and strengthens aligned-vs-shuffled signal,
> but it does not preserve enough class-specific evidence to beat raw content on
> balanced CORINE metrics or OSM metrics.

`content_with_evidence_card` improves CORINE raw accuracy and keeps strong
aligned-vs-shuffled deltas, but it does not clearly improve balanced accuracy or
macro-F1 over raw content. So the card can guide the model, but the raw article
still contains broader evidence that the compact representation loses.

The next best experiment should stop trying to replace raw text. A better next
step is to use the validated quality/relevance/spatial filters to select
candidate weak-supervision pairs for downstream Sentinel patch training or
evaluation. If we keep doing text-only diagnostics, the next useful variant
would be a retrieval-style prompt that asks the classifier to reason over raw
content while highlighting evidence sentences, not a shorter standalone card.
