# Source Code Overview

This directory contains the core logic for the GeoReset pipeline, which connects spatial land cover data with textual environmental descriptions.

## Files

- **[data_fetcher.py](file:///Users/noeflandre/georeset/src/data_fetcher.py)**: Manages loading and processing of Corine Land Cover (CLC) datasets using `geopandas`.
    - Handles CRS transformations (Lambert 93 to WGS84).
    - Extracts polygon centroids and bounding boxes.

- **[wiki_fetcher.py](file:///Users/noeflandre/georeset/src/wiki_fetcher.py)**: Fetches Wikipedia articles based on geographic coordinates.
    - Uses the Wikipedia `geosearch` API.
    - Implements a grid-based search to cover entire bounding boxes, as the API is limited to point-radius queries.
    - Includes retry logic and rate-limiting (polite headers).

- **[snapshot.py](file:///Users/noeflandre/georeset/src/snapshot.py)**: A utility script to provide a quick statistical overview of the loaded dataset (class distribution, polygon counts, etc.).

## Implementation Notes

- **Spatial Precision**: Centroid calculations are performed in Lambert 93 (meters) for accuracy before being converted back to WGS84 (degrees) for use with the Wikipedia API.
- **Coverage Strategy**: `WikiFetcher` uses an overlapping grid of circular searches (80% diameter step) to ensure no location is missed within a region's bounding box.
- **Modularity**: The fetchers are designed to be independent, allowing for easy extension to other data sources (e.g., OpenStreetMap or different land cover datasets).
- **Error Handling**: Wikipedia API calls include basic retry mechanisms to handle network instability during bulk downloads.
