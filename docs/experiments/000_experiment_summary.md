# GeoReset experiment summary

This document is a short report summarizing the GeoReset article-text classification work so far. The longer reports in this folder contain the detailed tables, this
note explains what we did, why we did it, what we found, and what should come
next. It is designed to stay concise, for further details please check the longer reports in this folder.

## What we are testing

We try to ask the following question:

Can geolocated (French for now) Wikipedia text help a local LLM infer land-cover labels
near the article location?

We tested this against two kinds of land-cover supervision:

- **CORINE Land Cover**, collapsed to level-2 classes such as arable land,
  permanent crops, pastures, forests, shrub/herbaceous vegetation, wetlands, and
  inland waters.
- **OpenStreetMap land-cover tags**, using a scoped set of `landuse`
  and `natural` polygon labels.

CORINE is treated as a single-label classification task (since each polygon has a single tag). OSM is treated as a
multi-label task, because one article point can intersect overlapping polygons,
and a single OSM polygon can contain both relevant `landuse` and `natural`
information.

## How the data was built

The CORINE polygons come from the 2018 regional land-cover shapefile for Alsace
available through DataGrandEst. The code loads the shapefile, derives level-2
labels from `code_18` (the numerical code for each label), and uses those polygons both for labels and for the
spatial-confidence analysis. We kept all labels except the ones associated with artificial surfaces since our interest lies across the agriculture, environment and so on.

The OSM polygons were fetched from Overpass. We only kept relevant
land-cover tags from OSM `landuse` and `natural` (these are lists of allowed tags we defined, please refer to the ppt or the video for details). These polygons are useful because they
are more local and more semantically varied than CORINE, but they also make the
task harder because the labels are multi-label and sometimes sparse and are non standards.

The Wikipedia articles were fetched from French Wikipedia by geosearch inside
the CORINE study area. For each article, we stored metadata such as `pageid`,
title, latitude, and longitude. We then fetched the full article extract by
pageid and filtered the dataset to articles that were spatially relevant to the
CORINE or OSM polygons. Basically an article is kept only if it falls in either an OSM polygon or a CORINE polygon or both.

For text inputs, we used three article representations:

- `summary`: a normal one-sentence summary where the place name may appear.
- `summary_no_place`: a one-sentence summary asking the summarizer not to
  mention the described place name.
- `content`: the raw article extract.

The two summary variants were generated once and then reused. We did not create
new summaries during the classifier reruns. The summarizer is a local
llama-cpp based article summarizer; using
`Qwen3.6-27B-Q4_0.gguf` with summarization temperature `0.7`. Classification
itself used temperature `0.0` and seed `42` throughout, so the experiments
compare models and inputs under a controlled decoding setup.

## What experiments were run

The first full experiment used `Qwen3.6-27B-Q4_0.gguf` as the classifier. It ran
12 jobs:

- 2 tasks: `corine_level2` and `osm`.
- 6 text sources: the three aligned sources mentioned above plus deterministic shuffled
  controls: `summary_shuffled`, `summary_no_place_shuffled`, and
  `content_shuffled`.

The shuffled controls are important. They keep the same target labels and the
same eligible article set, but they give each article the text from another
article. If aligned text beats shuffled text, the model is using article-specific
information rather than only label frequencies. It therefore would show that the text is having signal for the classifier.

After that, we also added a CORINE spatial-confidence experiment. This did not rerun
the LLM. Instead, it measured whether the area around each Wikipedia point was
actually dominated by the same CORINE label as the point-in-polygon label. We used 
buffers which were computed in `EPSG:2154`, not latitude/longitude, at 100 m, 250 m,
500 m, and 1000 m. The analysis used the full CORINE map, including artificial
classes, because nearby urban/artificial land is useful ambiguity evidence. The idea is that the area around that point might be dominated by other labels (for example a forest surrounded by villages). So we want to filter articles to keep the "purest" ones (i.e the ones for which the area surrounding them is mostly the same as the label associated with that article)

Finally, we repeated the same 12-run classifier protocol with
`gemma-4-31B-it-Q4_0.gguf` from `unsloth/gemma-4-31B-it-GGUF`. Nothing else
changed: same labels, prompts, text sources, shuffled logic, seed, temperature,
and spatial-confidence table.

## Main results

The Qwen first batch was complete and parse-clean: 1,251 CORINE examples and
275 OSM examples, with zero parse errors in the frozen outputs.

For CORINE, raw content was the strongest text source. It reached:

- accuracy: `0.293`
- balanced accuracy / macro recall: `0.281`
- macro-F1: `0.270`

Raw accuracy is not the best headline metric here. The CORINE dataset is
imbalanced, and the majority class, forests, already gives about `0.379`
accuracy if predicted everywhere. Under balanced metrics, however, the model is
clearly better than a majority baseline. When Qwen classifies based on the raw content of the article, it beats the majority
baseline by about `+0.170` balanced accuracy / macro recall and `+0.209`
macro-F1 in the spatial reevaluation.

The shuffled controls confirmed that the signal is text-linked. For Qwen CORINE
raw content, balanced accuracy was `0.281` with aligned content but only `0.067`
with shuffled content, a delta of `+0.213`.

The spatial-confidence experiment was one of the most important diagnostics.
Only about half of the 1,251 CORINE-labeled article points were clean examples
at 250 m under the rule `point_label_share_250m >= 0.8` (i.e 80% of the area surrounding the point at a radius of 250m is having the same label as the point ). Only about one third
were clean at 500 m. This means point-in-polygon labels are useful but noisy:
many Wikipedia coordinates sit near mixed land cover.

When we evaluated Qwen raw content only on the cleaner 250 m subset, CORINE balanced
accuracy improved from `0.281` to `0.352`, while shuffled content stayed low at
about `0.071`. That is the strongest positive result so far: article text is
more predictive when the spatial ground truth is more reliable.

For OSM, the task is harder. Exact-match accuracy is strict because the full
multi-label set must be correct. Qwen OSM raw content reached exact match `0.164`,
Jaccard `0.190`, micro-F1 `0.197`, and macro-F1 `0.158`. It did not beat the
most-frequent-label-set baseline on exact match or Jaccard, but it did beat
shuffled content clearly. On the high-confidence 250 m subset, Qwen OSM raw content
exact match rose to `0.225`, and shuffled content was only about `0.063`.

The Gemma rerun showed that the broad finding is robust to a second local LLM
family, but model behavior differs. Gemma improved CORINE raw accuracy and OSM
exact/micro metrics. For CORINE raw content, Gemma reached accuracy `0.357`, higher
than Qwen's `0.293`, but Qwen remained stronger on balanced CORINE behavior:
Qwen macro-F1 `0.270` versus Gemma `0.254`. On the high-confidence 250 m subset,
Gemma reached CORINE accuracy `0.455`, but Qwen kept better balanced accuracy
and macro-F1.

This tells us not to frame the result as one model simply winning. Gemma is more
majority-class aligned, especially for forests. Qwen is more balanced across
some minority CORINE classes.

## What this tells us

French Wikipedia text contains real land-cover signal, but the measured signal
depends strongly on spatial label reliability, class imbalance, metric choice,
and class semantics.

Some classes are naturally easier. Permanent crops, forests, water, and some
natural vegetation classes are more likely to be mentioned in text. Pastures,
generic arable land, and heterogeneous agricultural areas are much harder
because articles often do not describe them in a way that maps cleanly to
land-cover taxonomies.

The current summaries are also probably too generic and too short. Raw content usually beats
the one-sentence summaries, which suggests that summarization removed useful
landscape evidence.

We then tested deterministic no-place land-use evidence summaries. These short
summaries were parse-clean and text-linked, but they did not beat raw content as
a direct classifier input. The useful part turned out to be their metadata. When
we stratified the frozen Qwen and Gemma predictions by the evidence extractor's
`landcover_relevance`, raw content and generic summaries performed much better
on medium/high relevance articles. For example, Qwen CORINE raw-content balanced
accuracy improved from `0.281` on all articles to `0.374` on medium/high
relevance articles, and to `0.409` when medium/high relevance was combined with
the 250 m high-purity spatial subset. Gemma showed the same CORINE pattern,
going from `0.271` to `0.352`, and to `0.384` with the combined relevance +
spatial filter.

This changes the interpretation of the evidence extractor. It is not yet a good
replacement for raw content, but it is a useful quality filter for identifying
articles where the text is likely to contain real land-cover signal.

We then tested whether French Wikipedia category metadata could explain where
the classifier works. This was another analysis-only step: no LLM was rerun. We
fetched article categories/page properties, assigned a noisy primary article
type such as `natural_landscape`, `water_feature`, `agriculture_or_vineyard`,
`settlement_or_administrative`, or `other_or_unclear`, and recomputed the same
metrics by article type, relevance, and spatial confidence.

The category proxy is useful for interpretation, but it is not a stronger
filter than the evidence relevance score. Category metadata was incomplete:
646 of the 1,251 articles had no visible categories returned by the API, and 672
articles ended up in `other_or_unclear`. Agriculture/vineyard, water, and
natural-landscape pages were usually medium/high relevance, which makes sense.
But the best balanced CORINE signal still came from combining relevance and
spatial confidence, not from article type alone. In fact, medium/high relevance
inside the large `other_or_unclear` bucket still produced strong aligned-vs-
shuffled deltas. This means the evidence extractor finds useful land-cover
signal even when category metadata is missing or too vague.

We then added a deterministic supervision-quality pass (`008_supervision_quality_score`)
that assigns each article a numeric score and quality bin from five terms:
evidence relevance, spatial consistency, evidence density, article-type prior, and
uncertainty penalty. This is an analysis-only scoring step; no LLM rerun is done.

The same frozen predictions were then re-profiled by quality and recommended-use
subsets:

- `quality_low`, `quality_medium`, `quality_high`, and `quality_very_high`
- `exclude`, `use_for_training`, `use_for_evaluation_only`, and `inspect_manually`

Conservative precedence makes `exclude` override other suggestions when the score is
low or when relevance is `none`/uncertainty is `high`.

## What to do next

The next useful step is not to replace raw content with an even shorter summary.
The evidence-summary experiment suggests that compression is too lossy. A better
next direction is to use the evidence metadata as a filter or weight, then test a
richer but still compact evidence representation: for example a small evidence
card with several factual bullets and explicit evidence types. Article-type
metadata should remain a diagnostic context variable, not the main filter by
itself, unless a later Wikidata/entity-type version proves cleaner than the
current category proxy.

At this point, the strongest result is that geolocated Wikipedia text contains
real land-cover signal when both the spatial label and the article relevance are
credible. The bottleneck is no longer only the classifier model; it is the
quality of the supervision and the relevance of the text.

We then ran `article_text_evidence_card_v1`. This did not generate another LLM
summary. Instead, it deterministically turned existing evidence metadata,
article-type metadata, CORINE spatial confidence, and quality scores into a
compact French evidence card. We tested the card alone and the card prepended to
raw article content, using Qwen only.

The evidence card was better than the previous short `landuse_evidence_summary`
as a compact text representation. On all CORINE articles, balanced accuracy went
from `0.184` for `landuse_evidence_summary` to `0.224` for `evidence_card`, and
macro-F1 went from `0.203` to `0.226`. The card also had clear aligned-vs-
shuffled signal.

But raw content remained the strongest direct input. For CORINE, raw content
still had better macro-F1 (`0.270`) than `content_with_evidence_card` (`0.260`)
and `evidence_card` (`0.226`). `content_with_evidence_card` improved raw
accuracy (`0.337` versus `0.293` for content), but not the main balanced metrics.
For OSM, raw content also stayed best on Jaccard and micro-F1. The conclusion is
that cards organize useful metadata and are better than the previous compressed
summary, but they still lose too much class-specific evidence to replace raw
Wikipedia text.

The next useful step is therefore not another shorter text representation. It is
to use the relevance, spatial-confidence, and quality-score diagnostics to select
candidate weak-supervision pairs for downstream Sentinel patch training or
evaluation. If we continue with text-only diagnostics, the more promising
direction is a retrieval-style classifier prompt over raw content with highlighted
evidence sentences, not a standalone compact card.

We then tested this first highlighted-content variant in
`010_evidence_highlighted_content`, without running any new summarizer. The new
source prepends a deterministic evidence-highlight block to the raw article
content. It was run with both Qwen and Gemma on CORINE and OSM, with aligned and
shuffled controls.

The highlighted content was parse-clean and clearly text-linked: aligned
highlighted text beat shuffled highlighted text for both models and both tasks.
For example, CORINE balanced-accuracy deltas were about `+0.183` for Qwen and
`+0.184` for Gemma on all articles, and became larger on the combined
medium/high relevance plus 250 m spatial-purity subset.

But highlighted content still did not replace raw content. For Qwen CORINE, raw
content had balanced accuracy `0.281` and macro-F1 `0.270`, while highlighted
content had `0.273` and `0.256`. For Gemma CORINE, raw content had `0.271` and
`0.254`, while highlighted content had `0.259` and `0.253`. For OSM, raw content
remained stronger on exact match, micro-F1, and Jaccard for both models.
Highlighted content did beat the short land-use evidence summary on CORINE, so
it is a better representation than that compressed summary, but it is not better
than the full article.

This reinforces the current direction: the strongest value is in selecting
reliable article-location pairs using relevance and spatial diagnostics, not in
compressing or prepending metadata to the article text. The next major step
should therefore use these quality filters to build downstream weak-supervision
datasets for Sentinel patch training or evaluation.

We then ran `011_retrieved_evidence_windows`, which tested a stricter
retrieval-style text source. Instead of prepending metadata, the builder selects
raw Wikipedia sentences that match existing land-cover evidence, adds nearby
context sentences, and compares that to sentence-only, random-window, no-place,
and shuffled controls. The artifact covers all 1,251 articles; evidence was
matched in 752 articles, with complete fallback coverage for the rest.

The result is useful but not a reversal of the previous conclusion. Retrieved
windows are clearly text-linked and better than random windows for CORINE. On
all CORINE examples, Qwen retrieved windows reached balanced accuracy `0.266`
versus `0.254` for random windows and `0.087` for shuffled retrieved windows.
Gemma retrieved windows reached `0.245` versus `0.214` for random windows and
`0.077` for shuffled retrieved windows. But full raw content still remained
stronger: Qwen raw content was `0.281`, and Gemma raw content was `0.271`.

On the best quality-plus-spatial subset, retrieved windows were competitive but
still below raw content. For example, Gemma reached CORINE balanced accuracy
`0.358` with retrieved windows and `0.362` with no-place retrieved windows, while
raw content reached `0.375`. Qwen reached `0.341` with retrieved windows versus
`0.394` for raw content. OSM did not benefit from retrieval windows; raw content
remained best on Jaccard for both models.

This confirms the research direction: retrieved windows are a strong compact
diagnostic representation and a possible prompt-cost reduction source, but not a
replacement for raw content. The next major experiment should freeze a
high-confidence weak-supervision dataset using relevance, quality score, spatial
confidence, and model agreement, then test it in downstream Sentinel patch
training or evaluation.

We then ran that first downstream image test in
`012_clip_linear_probe_weak_labels`. The experiment used the existing quality
signals to build four CORINE weak-label training tiers, fetched 798 Sentinel-2
RGB patches from Planetary Computer, embedded them with frozen
`openai/clip-vit-base-patch32` image features, and trained a simple NumPy linear
probe. The fixed strict evaluation split had 35 examples, 5 per class.

The best tier was the broad `all` training set: accuracy `0.600`, balanced
accuracy `0.600`, and macro-F1 `0.586`. The `spatial_only` tier was close,
with accuracy `0.571`, balanced accuracy `0.571`, and macro-F1 `0.589`. The
stricter `quality_spatial` and `text_spatial_agreement` tiers were worse,
falling to balanced accuracy `0.514` and `0.400`. A true out-of-the-box
zero-shot CLIP baseline on the same evaluation split reached only accuracy
`0.200`, balanced accuracy `0.200`, and macro-F1 `0.224`, so the weakly
supervised linear probe adds a large gain over generic CLIP alignment.

The important finding is that the strict filters are too expensive as hard
filters for downstream image training at this data size. They remove noise, but
they also remove too many rare-class examples. For the next image experiment,
quality signals should be used as soft weights, sampling controls, or evaluation
strata rather than as a hard training-only gate.

We then ran `013_subset_randomization_controls`, an analysis-only reviewer-proofing
experiment. It reused frozen Qwen and Gemma predictions plus existing relevance,
spatial-confidence, article-type, and quality-score metadata. No LLM, prompt,
label, spatial recomputation, or GPU job was rerun.

The purpose was to test whether the earlier filtered-subset improvements survive
two Monte Carlo controls: random subsets with the same number of examples, and
random subsets with the same target distribution. The comparison universe was
kept explicit: same parent experiment, model, task, text source, and
subset-specific metadata availability.

The main result is that CORINE relevance and spatial-confidence claims are much
stronger after this control. For raw content, Qwen medium/high relevance reached
balanced accuracy `0.374` versus target-matched random mean `0.280`; the combined
medium/high relevance plus 250 m spatial-purity subset reached `0.409` versus
`0.281`. Gemma showed the same pattern, with `0.352` versus `0.271` for
medium/high relevance and `0.384` versus `0.272` for the combined subset.

For OSM the result is more cautious. Qwen raw content still has strong evidence
on the combined relevance plus spatial subset: Jaccard `0.291` versus
target-matched random mean `0.231`, with exact match `0.261`. But other OSM
rows beat same-size controls without clearly beating target-matched controls,
and Gemma OSM content is mostly not distinguishable from target-matched random
subsets. Rows with fewer than 30 examples, such as OSM
`recommended_use_evaluation_only`, are marked `unstable_small_n=true` and should
be treated as diagnostic only.

This rephrases the strongest text result: for CORINE, Wikipedia text is
predictive when article relevance and spatial label reliability are credible,
and that effect is not explained away by sample size or class composition. For
OSM, the evidence remains useful but should be framed more carefully because
target composition and small support explain more of the apparent gains.

Experiment `014_quality_weighted_multiscale_image_probe` implements the
follow-up to the first image probe. It keeps the broad weak-label training
coverage that worked best in Experiment 012, but turns quality, relevance,
spatial purity, and Qwen/Gemma agreement into soft sample weights instead of
hard filters. It also tests physical Sentinel-2 crop scale explicitly.

The first MVP attempt exposed an important implementation bug: the multiscale
Sentinel patch fetcher passed WGS84 longitude/latitude directly to
`rasterio.dataset.index()`, even though Sentinel assets are stored in projected
raster CRSs. With `boundless=True`, this silently produced all-black crops. That
invalidated the first `0.143` image-probe result.

After fixing the CRS transform and adding patch validation, the corrected MVP
used frozen `openai/clip-vit-base-patch32` embeddings on two Sentinel-2 RGB crop
sizes: 320 m and 2240 m, both resized to 224 x 224. The validation artifacts now
show 1,251 non-black patches per window, source-pixel sizes of 32 and 224,
plausible pixel means and variances, and non-identical 320 m versus 2240 m
arrays.

The corrected result is positive. Zero-shot CLIP reached only `0.200` accuracy
and supported balanced accuracy, with macro-F1 `0.152` at 320 m and `0.224` at
2240 m. The trained linear probe reached `0.686` strict supported balanced
accuracy at both 320 m and 2240 m. At 320 m, `text_agreement_soft_weighted`
clearly improved over `all_unweighted` (`0.686` versus `0.543` balanced
accuracy). At 2240 m, however, `all_unweighted` tied the best soft-weighted
policy on balanced accuracy (`0.686`) and was only slightly lower on macro-F1
(`0.677` versus `0.685`).

The hard-filter lesson from Experiment 012 still holds. Hard tiers such as
`quality_spatial_hard` and `agreement_hard` used only 286 and 152 training rows
and stayed below broad training. The strongest supported conclusion is not that
soft weighting always wins; it is that broad training coverage is essential, and
quality/agreement signals are safer as weights than as aggressive hard gates.
Wider 2240 m context was at least competitive with 320 m and slightly stronger
on macro-F1, but the MVP is still too small to declare a definitive scale
winner.
