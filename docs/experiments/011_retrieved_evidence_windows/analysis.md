# Retrieved Evidence Windows Experiment

## Question

The previous highlighted-content experiment showed that prepending deterministic
metadata to raw article text is text-linked, but still weaker than raw content.
This experiment tests a narrower question:

Can we keep only the raw article sentences that contain land-cover evidence,
plus a small amount of local context, and preserve most of the useful signal?

The motivation is different from the earlier compact cards and generated
evidence summaries. Here, the classifier sees original Wikipedia text, not a
rewritten summary. If this works, it gives a cheaper and cleaner candidate text
source for weak-supervision datasets.

## Method

The experiment builds `article_retrieved_evidence_windows.json` deterministically
from the existing article content and evidence metadata.

For each article, it creates:

- `retrieved_evidence_windows`: matched evidence sentences plus neighboring
  context sentences.
- `retrieved_evidence_sentences_only`: matched evidence sentences without
  context.
- `random_sentence_windows`: deterministic random sentence windows with the same
  article-level budget.
- `retrieved_evidence_windows_no_place`: the retrieved windows with article-title
  variants masked.
- `retrieved_evidence_windows_shuffled`: an aligned-vs-shuffled control built by
  the classifier source loader.

The builder uses exact evidence-sentence matching first, then a small typed
keyword fallback for known evidence types. It does not inject labels, model
predictions, coordinates, spatial scores, or quality scores into the classifier
text.

The final artifact has 1,251 article records. Evidence text was matched in 752
articles, with a mean of 4.948 matched evidence sentences and 10.297 retrieved
sentences after adding context. Articles without matched evidence fall back to
the first sentence so coverage remains complete and comparable.

## Runs

The classifier protocol matched earlier text-source experiments:

- Models: `Qwen3.6-27B-Q4_0.gguf` and `gemma-4-31B-it-Q4_0.gguf`.
- Tasks: `corine_level2` and `osm`.
- Temperature: `0.0`.
- Seed: `42`.
- CORINE examples: 1,251 per run.
- OSM examples: 275 per run.

All 20 GPU jobs completed. Every prediction file has the expected row count and
zero parse errors.

## Main Results

On all CORINE examples, retrieved evidence windows were close to raw content but
did not beat it. Qwen raw content reached balanced accuracy `0.281`, while Qwen
retrieved windows reached `0.266`. Gemma raw content reached `0.271`, while
Gemma retrieved windows reached `0.245`.

The retrieval signal is real. Retrieved windows beat random windows on CORINE
for both models:

- Qwen: `0.266` versus `0.254` balanced accuracy.
- Gemma: `0.245` versus `0.214` balanced accuracy.

They also beat shuffled retrieved windows by a wide margin:

- Qwen: `0.266` versus `0.087` balanced accuracy.
- Gemma: `0.245` versus `0.077` balanced accuracy.

The strongest subset-level result remains the combined quality and spatial
filter. On `quality_high_or_very_high_and_spatial_250m_ge_0.8`, retrieved windows
are competitive but still below raw content:

| model | source | n | accuracy | balanced accuracy | macro-F1 |
|---|---:|---:|---:|---:|---:|
| Gemma | content | 388 | 0.629 | 0.375 | 0.334 |
| Gemma | retrieved windows | 388 | 0.531 | 0.358 | 0.305 |
| Gemma | retrieved no-place | 388 | 0.536 | 0.362 | 0.318 |
| Qwen | content | 388 | 0.490 | 0.394 | 0.332 |
| Qwen | retrieved windows | 388 | 0.451 | 0.341 | 0.296 |
| Qwen | retrieved no-place | 388 | 0.423 | 0.340 | 0.282 |

On OSM, retrieved windows do not improve the main metrics. Raw content remains
best on Jaccard for both models:

- Qwen: content `0.190`, retrieved windows `0.157`, no-place `0.163`.
- Gemma: content `0.252`, retrieved windows `0.208`, no-place `0.207`.

This is consistent with earlier results: OSM labels are more local, sparse, and
multi-label, while the retrieval heuristic is based on broad land-cover evidence
sentences.

## Interpretation

Retrieved evidence windows are the best compact raw-text representation tested
so far for CORINE. They are better than random windows, sentence-only retrieval
is usually weaker than adding context, and shuffled controls stay near chance.
That means the retrieval mechanism is selecting meaningful article-specific
evidence.

But the experiment also rejects the stronger hypothesis that retrieved windows
should replace full raw content. Full articles still carry useful cues outside
the matched evidence sentences: place type, local context, surrounding prose,
and indirect landscape descriptions.

The no-place variant is informative. Deterministic article-title scrubbing did not
collapse performance; for Gemma it slightly improved CORINE balanced accuracy.
So the signal is not only memorized place-name association. It is mostly carried
by the article text itself.

The research conclusion is therefore:

- Use retrieved windows as a compact diagnostic source and possible prompt-cost
  reduction tool.
- Do not replace raw content with retrieved windows for the main weak-label
  selection pipeline.
- Keep relevance, quality score, and spatial confidence as the primary filters
  for selecting article-location pairs.

## Artifacts

Generated data and analysis artifacts:

- `data/wiki/article_retrieved_evidence_windows.json`
- `data/experiments/011_retrieved_evidence_windows/article_text_retrieved_evidence_windows_v1__qwen3_6_27b_q4_0/`
- `data/experiments/011_retrieved_evidence_windows/article_text_retrieved_evidence_windows_v1__gemma4_31b_it_q4_0/`
- `data/experiments/011_retrieved_evidence_windows/retrieved_evidence_windows_comparison_v1/`

Code paths:

- `src/georeset_wiki_landcover/text/retrieved_evidence_windows.py`
- `src/georeset_wiki_landcover/cli/data/build_retrieved_evidence_windows.py`
- `src/georeset_wiki_landcover/cli/analysis/evaluate_retrieved_evidence_windows_experiment.py`

## Next Step

The next experiment should move away from more text-source compression. The
evidence is now stable across cards, highlighted content, and retrieved windows:
raw content remains strongest, while deterministic evidence artifacts are most
useful for filtering, diagnostics, and cost reduction.

The next research-grade step is to freeze a candidate weak-supervision dataset
using quality, spatial confidence, relevance, and model agreement, then evaluate
whether those selected article-location pairs support downstream Sentinel patch
training or held-out evaluation.
