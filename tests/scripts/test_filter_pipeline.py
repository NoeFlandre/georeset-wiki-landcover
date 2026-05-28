"""Tests for filter_pipeline script."""

import json
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import box

from georeset_wiki_landcover.cli.data.filter_pipeline import (
    filter_articles_by_polygons,
    filter_osm_by_corine,
    filter_pipeline,
)


def _write_corine_fixture(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    corine = gpd.GeoDataFrame(
        {
            "code_18": ["311", "112"],
            "geometry": [
                box(7.0, 48.0, 8.0, 49.0),
                box(6.0, 47.0, 6.5, 47.5),
            ],
        },
        crs="EPSG:4326",
    )
    corine.to_file(path)
    return path


class TestFilterPipeline:
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create a temporary data directory with sample files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        corine_dir = data_dir / "corine"
        corine_dir.mkdir()
        (corine_dir / "bounds.json").write_text(
            json.dumps({"min_lon": 7.0, "min_lat": 48.0, "max_lon": 8.0, "max_lat": 49.0})
        )
        _write_corine_fixture(
            corine_dir / "alsace_corine_land_use_2018" / "occupation_sol_2018.shp"
        )

        wiki_dir = data_dir / "wiki"
        wiki_dir.mkdir()

        osm_dir = data_dir / "osm"
        osm_dir.mkdir()

        dist_dir = data_dir / "distribution"
        dist_dir.mkdir()

        maps_dir = data_dir / "maps"
        maps_dir.mkdir()

        return data_dir

    def test_filter_removes_articles_outside_corine(self, temp_data_dir, monkeypatch):
        """Should remove wiki articles whose coordinates are not in any filtered CORINE polygon."""
        from georeset_wiki_landcover.fetchers.data_fetcher import DataFetcher

        corine_path = (
            temp_data_dir / "corine" / "alsace_corine_land_use_2018" / "occupation_sol_2018.shp"
        )
        monkeypatch.setattr(
            "georeset_wiki_landcover.cli.data.filter_pipeline.DataFetcher",
            lambda: DataFetcher(data_path=str(corine_path)),
        )

        wiki_articles = [
            {"pageid": 1, "lat": 48.5, "lon": 7.5},  # inside CORINE
            {"pageid": 2, "lat": 48.5, "lon": 9.5},  # outside CORINE
        ]
        wiki_path = temp_data_dir / "wiki" / "wiki_articles.json"
        wiki_path.write_text(json.dumps(wiki_articles))

        article_contents_path = temp_data_dir / "wiki" / "article_contents.json"
        article_contents_path.write_text(
            json.dumps(
                {
                    "1": {"title": "In", "content": "Content in", "url": "http://in"},
                    "2": {"title": "Out", "content": "Content out", "url": "http://out"},
                }
            )
        )

        article_summaries_path = temp_data_dir / "wiki" / "article_summaries.json"
        article_summaries_path.write_text(
            json.dumps(
                {
                    "1": {
                        "title": "In",
                        "content": "Content in",
                        "url": "http://in",
                        "summary": "Sum in",
                    },
                    "2": {
                        "title": "Out",
                        "content": "Content out",
                        "url": "http://out",
                        "summary": "Sum out",
                    },
                }
            )
        )
        article_summaries_no_place_path = temp_data_dir / "wiki" / "article_summaries_no_place.json"
        article_summaries_no_place_path.write_text(
            json.dumps(
                {
                    "1": {"title": "In", "summary": "No place in"},
                    "2": {"title": "Out", "summary": "No place out"},
                }
            )
        )

        osm_path = temp_data_dir / "osm" / "osm_project_polygons.geojson"
        osm_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"osm_id": "way/1"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [7.4, 48.4],
                                        [7.6, 48.4],
                                        [7.6, 48.6],
                                        [7.4, 48.6],
                                        [7.4, 48.4],
                                    ]
                                ],
                            },
                        }
                    ],
                }
            )
        )

        distribution_path = temp_data_dir / "distribution" / "osm_corine_distribution.csv"
        distribution_path.write_text("osm_id,class_label,area,share\nway/1,311,1000,1.0\n")

        maps_path = temp_data_dir / "maps" / "osm_corine_polygons.html"
        maps_path.write_text("<html>old map</html>")

        with patch.dict("os.environ", {"DATA_DIR": str(temp_data_dir)}):
            filter_pipeline(
                wiki_articles_path=str(wiki_path),
                article_contents_path=str(article_contents_path),
                article_summaries_path=str(article_summaries_path),
                article_summaries_no_place_path=str(article_summaries_no_place_path),
                osm_polygons_path=str(osm_path),
                distribution_csv_path=str(distribution_path),
                map_articles_path=str(temp_data_dir / "maps" / "corine_with_articles.html"),
                map_osm_path=str(maps_path),
            )

        with open(wiki_path) as f:
            filtered_articles = json.load(f)
        with open(article_contents_path) as f:
            filtered_contents = json.load(f)
        with open(article_summaries_path) as f:
            filtered_summaries = json.load(f)
        with open(article_summaries_no_place_path) as f:
            filtered_no_place_summaries = json.load(f)

        assert len(filtered_articles) == 1
        assert filtered_articles[0]["pageid"] == 1
        assert "1" in filtered_contents
        assert "2" not in filtered_contents
        assert "1" in filtered_summaries
        assert "2" not in filtered_summaries
        assert "1" in filtered_no_place_summaries
        assert "2" not in filtered_no_place_summaries

    def test_missing_wiki_articles_fails_before_pruning_content(self, temp_data_dir):
        missing_wiki_path = temp_data_dir / "wiki" / "missing_wiki_articles.json"
        article_contents_path = temp_data_dir / "wiki" / "article_contents.json"
        article_contents_path.write_text(json.dumps({"1": {"content": "keep me"}}))

        with pytest.raises(FileNotFoundError, match="wiki articles"):
            filter_pipeline(
                wiki_articles_path=str(missing_wiki_path),
                article_contents_path=str(article_contents_path),
                article_summaries_path=str(temp_data_dir / "wiki" / "article_summaries.json"),
                article_summaries_no_place_path=str(
                    temp_data_dir / "wiki" / "article_summaries_no_place.json"
                ),
                osm_polygons_path=str(temp_data_dir / "osm" / "missing.geojson"),
                distribution_csv_path=str(
                    temp_data_dir / "distribution" / "osm_corine_distribution.csv"
                ),
                map_articles_path=str(temp_data_dir / "maps" / "corine_with_articles.html"),
                map_osm_path=str(temp_data_dir / "maps" / "osm_corine_polygons.html"),
            )

        assert json.loads(article_contents_path.read_text()) == {"1": {"content": "keep me"}}

    def test_dry_run_reports_without_mutating_artifacts(self, temp_data_dir, monkeypatch):
        from georeset_wiki_landcover.fetchers.data_fetcher import DataFetcher

        corine_path = (
            temp_data_dir / "corine" / "alsace_corine_land_use_2018" / "occupation_sol_2018.shp"
        )
        monkeypatch.setattr(
            "georeset_wiki_landcover.cli.data.filter_pipeline.DataFetcher",
            lambda: DataFetcher(data_path=str(corine_path)),
        )

        wiki_path = temp_data_dir / "wiki" / "wiki_articles.json"
        wiki_path.write_text(json.dumps([{"pageid": 1, "lat": 48.5, "lon": 7.5}]))
        contents_path = temp_data_dir / "wiki" / "article_contents.json"
        contents_path.write_text(json.dumps({"1": {"content": "keep"}, "2": {"content": "stale"}}))
        summaries_path = temp_data_dir / "wiki" / "article_summaries.json"
        summaries_path.write_text(json.dumps({"1": {"summary": "keep"}, "2": {"summary": "stale"}}))
        no_place_path = temp_data_dir / "wiki" / "article_summaries_no_place.json"
        no_place_path.write_text(
            json.dumps({"1": {"summary": "keep no place"}, "2": {"summary": "stale no place"}})
        )
        osm_path = temp_data_dir / "osm" / "osm_project_polygons.geojson"
        osm_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"osm_id": "way/1"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [7.4, 48.4],
                                        [7.6, 48.4],
                                        [7.6, 48.6],
                                        [7.4, 48.6],
                                        [7.4, 48.4],
                                    ]
                                ],
                            },
                        }
                    ],
                }
            )
        )

        before = {
            path: path.read_text()
            for path in [wiki_path, contents_path, summaries_path, no_place_path, osm_path]
        }

        filter_pipeline(
            wiki_articles_path=str(wiki_path),
            article_contents_path=str(contents_path),
            article_summaries_path=str(summaries_path),
            article_summaries_no_place_path=str(no_place_path),
            osm_polygons_path=str(osm_path),
            distribution_csv_path=str(
                temp_data_dir / "distribution" / "osm_corine_distribution.csv"
            ),
            map_articles_path=str(temp_data_dir / "maps" / "corine_with_articles.html"),
            map_osm_path=str(temp_data_dir / "maps" / "osm_corine_polygons.html"),
            dry_run=True,
        )

        assert {path: path.read_text() for path in before} == before


class TestFilterCorineStep:
    """Tests for CORINE filtering step."""

    def test_exclude_artificial_surface_polygons(self, tmp_path):
        """Artificial surface polygons (code starting with 1) should be excluded."""
        from georeset_wiki_landcover.fetchers.data_fetcher import DataFetcher

        data_path = _write_corine_fixture(tmp_path / "corine" / "occupation_sol_2018.shp")
        fetcher = DataFetcher(data_path=str(data_path))
        gdf = fetcher.load_data(exclude_artificial=True)
        artificial = gdf[gdf["code_18"].str.startswith("1")]
        assert len(artificial) == 0


class TestFilterOsmStep:
    """Tests for cascading artificial surface exclusion into OSM polygons."""

    def test_filter_osm_keeps_polygons_overlapping_filtered_corine(self):
        """OSM polygon spanning artificial raw CORINE and non-artificial filtered CORINE is kept.
        OSM 'mixed' covers x=0..10, overlapping artificial 112 at x=0..2 AND natural 311 at x=2..10.
        OSM 'natural' covers x=20..30, overlapping only natural 211.
        Result: both 'mixed' and 'natural' must be kept."""
        osm = gpd.GeoDataFrame(
            {
                "osm_id": ["mixed", "natural"],
                "geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)],
            },
            crs="EPSG:3857",
        )
        full_corine = gpd.GeoDataFrame(
            {
                "code_18": ["112", "311", "211"],
                "geometry": [box(0, 0, 2, 10), box(2, 0, 10, 10), box(20, 0, 30, 10)],
            },
            crs="EPSG:3857",
        )
        filtered_corine = full_corine[~full_corine["code_18"].str.startswith("1")].copy()

        filtered = filter_osm_by_corine(osm, filtered_corine)

        assert filtered["osm_id"].tolist() == ["mixed", "natural"]


class TestFilterArticlesStep:
    @pytest.fixture
    def corine_gdf(self):
        return gpd.GeoDataFrame(
            {
                "code_18": ["311"],
                "geometry": [box(7.0, 48.0, 7.1, 48.1)],
            },
            crs="EPSG:4326",
        )

    @pytest.fixture
    def osm_gdf(self):
        return gpd.GeoDataFrame(
            {
                "osm_id": ["way/1"],
                "geometry": [box(7.05, 48.05, 7.15, 48.15)],
            },
            crs="EPSG:4326",
        )

    def test_article_in_both_is_kept_once(self, corine_gdf, osm_gdf):
        # Point (7.075, 48.075) is inside OSM interior — not on boundary at (7.05, 48.05)
        articles = [{"pageid": 1, "lat": 48.075, "lon": 7.075}]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_gdf)
        assert len(result) == 1
        assert result[0]["pageid"] == 1

    def test_article_in_corine_only_not_in_osm_is_kept(self, corine_gdf, osm_gdf):
        articles = [{"pageid": 2, "lat": 48.02, "lon": 7.02}]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_gdf)
        assert len(result) == 1
        assert result[0]["pageid"] == 2

    def test_article_in_osm_only_not_in_corine_is_kept(self, corine_gdf, osm_gdf):
        osm_only = gpd.GeoDataFrame(
            {
                "osm_id": ["way/2"],
                "geometry": [box(8.0, 48.0, 8.1, 48.1)],
            },
            crs="EPSG:4326",
        )
        articles = [{"pageid": 3, "lat": 48.05, "lon": 8.05}]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_only)
        assert len(result) == 1
        assert result[0]["pageid"] == 3

    def test_article_on_corine_boundary_is_kept_by_intersects_policy(self, osm_gdf):
        corine = gpd.GeoDataFrame(
            {
                "code_18": ["211", "311"],
                "geometry": [box(7.0, 48.0, 7.1, 48.1), box(7.1, 48.0, 7.2, 48.1)],
            },
            crs="EPSG:4326",
        )
        articles = [{"pageid": 12, "lat": 48.05, "lon": 7.1}]

        result = filter_articles_by_polygons(articles, corine, osm_gdf.iloc[:0].copy())

        assert [article["pageid"] for article in result] == [12]

    def test_article_on_osm_boundary_is_kept_by_intersects_policy(self, corine_gdf):
        osm = gpd.GeoDataFrame(
            {
                "osm_id": ["way/1"],
                "geometry": [box(8.0, 48.0, 8.1, 48.1)],
            },
            crs="EPSG:4326",
        )
        articles = [{"pageid": 13, "lat": 48.05, "lon": 8.0}]

        result = filter_articles_by_polygons(articles, corine_gdf.iloc[:0].copy(), osm)

        assert [article["pageid"] for article in result] == [13]

    def test_article_in_neither_is_dropped(self, corine_gdf, osm_gdf):
        articles = [{"pageid": 4, "lat": 50.0, "lon": 10.0}]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_gdf)
        assert len(result) == 0

    def test_malformed_coords_are_dropped(self, corine_gdf, osm_gdf):
        articles = [
            {"pageid": 5, "lat": None, "lon": 7.05},
            {"pageid": 6, "lat": 48.05, "lon": "bad"},
            {"pageid": 7, "lat": 48.05},  # missing lon
        ]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_gdf)
        assert len(result) == 0

    def test_duplicate_pageids_deduplicated(self, corine_gdf, osm_gdf):
        articles = [
            {"pageid": 8, "lat": 48.075, "lon": 7.075},
            {"pageid": 8, "lat": 48.075, "lon": 7.075},
        ]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_gdf)
        assert len(result) == 1

    def test_preserves_source_order(self, corine_gdf, osm_gdf):
        osm_only = gpd.GeoDataFrame(
            {
                "osm_id": ["way/2"],
                "geometry": [box(8.0, 48.0, 8.1, 48.1)],
            },
            crs="EPSG:4326",
        )
        articles = [
            {"pageid": 10, "lat": 48.05, "lon": 8.05},
            {"pageid": 11, "lat": 48.02, "lon": 7.02},
        ]
        result = filter_articles_by_polygons(articles, corine_gdf, osm_only)
        assert result[0]["pageid"] == 10
        assert result[1]["pageid"] == 11
