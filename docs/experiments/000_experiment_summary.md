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
