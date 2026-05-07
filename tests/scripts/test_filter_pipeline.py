"""Tests for filter_pipeline script."""

import json
import tempfile
from unittest.mock import patch

import pytest

from scripts.data.filter_pipeline import filter_pipeline


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

        wiki_dir = data_dir / "wiki"
        wiki_dir.mkdir()

        osm_dir = data_dir / "osm"
        osm_dir.mkdir()

        dist_dir = data_dir / "distribution"
        dist_dir.mkdir()

        maps_dir = data_dir / "maps"
        maps_dir.mkdir()

        return data_dir

    def test_filter_removes_articles_outside_corine(self, temp_data_dir):
        """Should remove wiki articles whose coordinates are not in any filtered CORINE polygon."""
        wiki_articles = [
            {"pageid": 1, "lat": 48.5, "lon": 7.5},   # inside CORINE
            {"pageid": 2, "lat": 48.5, "lon": 9.5},   # outside CORINE
        ]
        wiki_path = temp_data_dir / "wiki" / "wiki_articles.json"
        wiki_path.write_text(json.dumps(wiki_articles))

        article_contents_path = temp_data_dir / "wiki" / "article_contents.json"
        article_contents_path.write_text(json.dumps({
            "1": {"title": "In", "content": "Content in", "url": "http://in"},
            "2": {"title": "Out", "content": "Content out", "url": "http://out"},
        }))

        article_summaries_path = temp_data_dir / "wiki" / "article_summaries.json"
        article_summaries_path.write_text(json.dumps({
            "1": {"title": "In", "content": "Content in", "url": "http://in", "summary": "Sum in"},
            "2": {"title": "Out", "content": "Content out", "url": "http://out", "summary": "Sum out"},
        }))

        osm_path = temp_data_dir / "osm" / "osm_project_polygons.geojson"
        osm_path.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"osm_id": "way/1"},
                    "geometry": {"type": "Polygon", "coordinates": [[[7.4, 48.4], [7.6, 48.4], [7.6, 48.6], [7.4, 48.6], [7.4, 48.4]]]},
                }
            ]
        }))

        distribution_path = temp_data_dir / "distribution" / "osm_corine_distribution.csv"
        distribution_path.write_text("osm_id,class_label,area,share\nway/1,311,1000,1.0\n")

        maps_path = temp_data_dir / "maps" / "osm_corine_polygons.html"
        maps_path.write_text("<html>old map</html>")

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict("os.environ", {"DATA_DIR": str(temp_data_dir)}):
            filter_pipeline(
                wiki_articles_path=str(wiki_path),
                article_contents_path=str(article_contents_path),
                article_summaries_path=str(article_summaries_path),
                osm_polygons_path=str(osm_path),
                distribution_csv_path=str(distribution_path),
                map_html_path=str(maps_path),
                output_dir=tmpdir,
            )

        with open(wiki_path) as f:
            filtered_articles = json.load(f)
        with open(article_contents_path) as f:
            filtered_contents = json.load(f)
        with open(article_summaries_path) as f:
            filtered_summaries = json.load(f)

        assert len(filtered_articles) == 1
        assert filtered_articles[0]["pageid"] == 1
        assert "1" in filtered_contents
        assert "2" not in filtered_contents
        assert "1" in filtered_summaries
        assert "2" not in filtered_summaries


class TestFilterCorineStep:
    """Tests for CORINE filtering step."""

    def test_exclude_artificial_surface_polygons(self):
        """Artificial surface polygons (code starting with 1) should be excluded."""
        from src.fetchers.data_fetcher import DataFetcher

        fetcher = DataFetcher()
        gdf = fetcher.load_data(exclude_artificial=True)
        artificial = gdf[gdf["code_18"].str.startswith("1")]
        assert len(artificial) == 0
