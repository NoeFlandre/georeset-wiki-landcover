# Evidence-highlighted content experiment

## What changed

This experiment tested one narrow idea: keep the raw Wikipedia article, but
prepend a deterministic no-LLM evidence block built from metadata we already
had. The new text source is:

- `content_with_evidence_highlights`
- `content_with_evidence_highlights_shuffled`

The evidence block uses extracted no-place evidence sentences, relevance,
uncertainty, and evidence types. It deliberately does **not** include CORINE
targets, OSM targets, predicted labels, coordinates, point labels, dominant
labels, spatial purity values, or quality scores. The raw content is then
preserved after the block, so this source is not no-place: the original article
may still contain place names.

No summaries were regenerated. No LLM was used to build the highlights. The
classification protocol stayed fixed: seed `42`, temperature `0.0`, same labels,
same classifier prompts, and same shuffled-control logic.

## Artifacts checked

The new frozen outputs are:

- `data/experiments/article_text_evidence_highlights_v1__qwen3_6_27b_q4_0/`
- `data/experiments/article_text_evidence_highlights_v1__gemma4_31b_it_q4_0/`
- `data/experiments/evidence_highlights_comparison_v1/`

Both Qwen and Gemma were run on CORINE and OSM, aligned and shuffled. All eight
prediction files are parse-clean:

- Qwen CORINE aligned/shuffled: `1251/1251` ok
- Gemma CORINE aligned/shuffled: `1251/1251` ok
- Qwen OSM aligned/shuffled: `275/275` ok
- Gemma OSM aligned/shuffled: `275/275` ok

The numbers below are taken from
`data/experiments/evidence_highlights_comparison_v1/evidence_highlights_quality_subsets.csv`
and `evidence_highlights_shuffled_deltas.csv`.

## Main result

Evidence-highlighted content works in the sense that it is strongly
article-linked: aligned highlighted content beats shuffled highlighted content
for both models and both tasks.

On all articles:

| model | task | metric | aligned | shuffled | delta |
|---|---:|---:|---:|---:|---:|
| Qwen | CORINE | balanced accuracy | 0.273 | 0.090 | +0.183 |
| Gemma | CORINE | balanced accuracy | 0.259 | 0.075 | +0.184 |
| Qwen | OSM | Jaccard | 0.148 | 0.065 | +0.084 |
| Gemma | OSM | Jaccard | 0.207 | 0.091 | +0.116 |

So the highlighted source is not random noise. It preserves real text signal.

## Does it beat raw content?

Mostly no.

For CORINE on all articles:

| model | source | accuracy | balanced accuracy | macro-F1 |
|---|---|---:|---:|---:|
| Qwen | raw content | 0.293 | 0.281 | 0.270 |
| Qwen | highlighted content | 0.313 | 0.273 | 0.256 |
| Gemma | raw content | 0.357 | 0.271 | 0.254 |
| Gemma | highlighted content | 0.321 | 0.259 | 0.253 |

Qwen highlighted content improves raw accuracy but loses balanced accuracy and
macro-F1. Gemma highlighted content is slightly below raw content across the
main CORINE metrics.

For OSM on all articles:

| model | source | exact match | micro-F1 | macro-F1 | Jaccard |
|---|---|---:|---:|---:|---:|
| Qwen | raw content | 0.164 | 0.197 | 0.158 | 0.190 |
| Qwen | highlighted content | 0.102 | 0.180 | 0.116 | 0.148 |
| Gemma | raw content | 0.215 | 0.247 | 0.158 | 0.252 |
| Gemma | highlighted content | 0.153 | 0.218 | 0.150 | 0.207 |

Raw content remains the best direct input for OSM.

## Does it beat summaries and land-use evidence summaries?

The answer is mixed and model-dependent.

For CORINE on all articles, highlighted content is clearly better than the short
`landuse_evidence_summary` for both models:

- Qwen balanced accuracy: `0.273` vs `0.184`
- Qwen macro-F1: `0.256` vs `0.203`
- Gemma balanced accuracy: `0.259` vs `0.218`
- Gemma macro-F1: `0.253` vs `0.219`

It also beats the generic summaries for Gemma on CORINE, and is close to or
above generic summaries for Qwen depending on the metric.

For OSM, however, the picture is weaker. Qwen highlighted content is below raw
content, below `summary_no_place`, and below the short evidence summary on
Jaccard. Gemma highlighted content beats the short evidence summary, but still
loses to raw content and generic summaries.

## High-relevance and high-spatial subsets

The highlighted source improves when we filter to better supervision, but it
still usually does not beat raw content.

On the combined `relevance_medium_high_and_spatial_250m_ge_0.8` subset:

| model | task | source | n | main metric |
|---|---|---|---:|---:|
| Qwen | CORINE | raw content balanced accuracy | 321 | 0.409 |
| Qwen | CORINE | highlighted content balanced accuracy | 321 | 0.361 |
| Gemma | CORINE | raw content balanced accuracy | 321 | 0.384 |
| Gemma | CORINE | highlighted content balanced accuracy | 321 | 0.379 |
| Qwen | OSM | raw content Jaccard | 92 | 0.291 |
| Qwen | OSM | highlighted content Jaccard | 92 | 0.234 |
| Gemma | OSM | raw content Jaccard | 92 | 0.294 |
| Gemma | OSM | highlighted content Jaccard | 92 | 0.223 |

The best highlighted result is Gemma CORINE, where highlighted content nearly
matches raw content on the clean relevance+spatial subset. But the broader
pattern is still that raw content is stronger.

The good news is that shuffled highlighted content stays low in these subsets.
For example, on the combined relevance+spatial subset, highlighted CORINE
balanced-accuracy deltas are:

- Qwen: `0.361 - 0.080 = +0.281`
- Gemma: `0.379 - 0.073 = +0.307`

That confirms the highlighted representation remains text-linked even in the
cleaner subsets.

## Conclusion

The experiment gives a clear answer: deterministic evidence highlighting is a
better representation than the previous short land-use evidence summary for
CORINE, but it is not a better replacement for raw content.

The most likely reason is that the metadata-derived evidence block is useful
context, but prepending it changes the prompt budget and attention pattern
without adding enough new information. The raw article already contains the
landscape cues; compressing or front-loading them helps less than expected.

The current best scientific conclusion remains:

> Wikipedia text contains real land-cover signal, especially when article
> relevance and spatial label reliability are credible, but raw content is still
> the strongest direct classifier input.

## Recommended next step

Do not build another compact text representation next. The next most useful
experiment should move from text-only diagnostics toward using the quality and
spatial filters to select candidate weak-supervision pairs for downstream
Sentinel patch training/evaluation.

If we do one more text-only diagnostic, it should not prepend metadata. It
should test retrieval over raw content: classify from the top evidence-bearing
raw sentences plus a small amount of surrounding context, while keeping a
shuffled control. That would test whether the useful signal is localized in the
article or distributed across the full text.
