# GeoReset

This repository is part of the GeoReset project: https://geo-reset.sylvainlobry.com/

This codebase focuses on experiments around CORINE land-cover polygons, nearby
Wikipedia articles, and OpenStreetMap land-cover polygons for the same Alsace
bounds.

Code lives on GitHub: https://github.com/NoeFlandre/georeset

Generated and downloaded project data lives in the Hugging Face bucket:
https://huggingface.co/buckets/NoeFlandre/georeset

Sync the data locally with:

```bash
hf sync hf://buckets/NoeFlandre/georeset ./data
```

## Current Pipeline

- Choose a map (CORINE? EUNIS? LUCAS?) ==> CORINE
- Choose a region (Alsace?) ==> Alsace
- Choose the class to keep ==> Stick to a few number of classes (Level 1 or Level 2)
- Scrape polygons and their classes from the map
- Scrape relevant web pages with respect to these polygons
- Ask an LLM to summarize / rephrase these web pages
- Make sure there is no mention of the place itself in the resulting descriptions
- Give the rephrased descriptions to another LLM and ask it to classify the associated polygon (classes will be provided to the LLM)
- We evaluate the classification produced by the LLM using standard metrics like precision, recall and so on.

The goal here is to assess whether based solely on the text description an LLM is able to deduce the correct class of the polygon.

The data used was downloaded here: https://www.datagrandest.fr/geonetwork/srv/api/records/c0ccbf45-2620-4bde-93f8-869558e51d7e?language=fre

## Main Modules

- `src/data_fetcher.py`: loads the CORINE shapefile and exposes its bounds.
- `src/wiki_fetcher.py`: fetches French Wikipedia geosearch articles inside the CORINE bounds and optional polygon filter.
- `src/osm_fetcher.py`: fetches OSM polygons from Overpass for the same CORINE bounds.
- `src/corine_polygon_stats.py`: computes CORINE class area/share distributions inside OSM polygons.
- `src/map_visualizer.py`: writes Folium maps for visual checks.
- `src/run_corine_analysis.py`: orchestrates the OSM fetch, CORINE distribution CSV, and OSM/CORINE map.

## OSM Scope

OSM fetching is intentionally restricted to project-relevant land-cover tags.
It excludes dense or unrelated tags such as buildings, amenities, commercial,
industrial, residential, and leisure features.

Included `landuse` values:

```text
farmland, farmyard, meadow, orchard, vineyard, forest, allotments,
plant_nursery, greenhouse_horticulture, grass
```

Included `natural` values:

```text
wood, scrub, grassland, wetland, heath, water, bare_rock, sand, scree,
shingle, beach, mud
```

## Generated Artifacts

Generated artifacts are intentionally not tracked in Git. They should be synced
through the Hugging Face bucket above.

- `data/corine/`: CORINE shapefile and bounds
- `data/osm/osm_project_polygons.geojson`: full project-relevant OSM polygon set for the same CORINE bounds
- `data/distribution/osm_corine_distribution.csv`: area and share of CORINE classes inside each OSM polygon
- `data/maps/`: HTML map visualizations (CORINE + OSM, CORINE + Wikipedia articles)

## Commands

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest
```

Fetch Wikipedia articles:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.wiki_fetcher
```

Regenerate the CORINE + Wikipedia article map:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.map_visualizer
```

Fetch OSM polygons, compute CORINE distributions inside them, and generate the
separate CORINE + OSM polygon map:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.osm_corine_analysis
```
