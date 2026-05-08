import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.data.classify_articles import parse_args, prediction_fingerprint


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("GEORESET_MODEL_PATH", raising=False)
        args = parse_args([])
        assert args.task == "corine_level2"
        assert args.text_source == "summary"
        assert args.model_path == "Qwen3.6-27B-Q4_0.gguf"
        assert args.seed == 42
        assert args.temperature == 0.0
        assert args.limit is None

    def test_env_model_override(self, monkeypatch):
        monkeypatch.setenv("GEORESET_MODEL_PATH", "custom.gguf")
        args = parse_args(["--task", "osm"])
        assert args.model_path == "custom.gguf"

    def test_limit_option(self, monkeypatch):
        monkeypatch.delenv("GEORESET_MODEL_PATH", raising=False)
        args = parse_args(["--limit", "5"])
        assert args.limit == 5


class TestFingerprint:
    def test_same_inputs_same_fingerprint(self):
        fp1 = prediction_fingerprint(
            "corine_level2", "summary", "m.gguf", 42, 0.0, ["21", "31"]
        )
        fp2 = prediction_fingerprint(
            "corine_level2", "summary", "m.gguf", 42, 0.0, ["21", "31"]
        )
        assert fp1 == fp2

    def test_different_task_different_fingerprint(self):
        fp1 = prediction_fingerprint(
            "corine_level2", "summary", "m.gguf", 42, 0.0, ["21"]
        )
        fp2 = prediction_fingerprint("osm", "summary", "m.gguf", 42, 0.0, ["meadow"])
        assert fp1 != fp2

    def test_fingerprint_order_independent(self):
        fp1 = prediction_fingerprint(
            "corine_level2", "summary", "m.gguf", 42, 0.0, ["31", "21"]
        )
        fp2 = prediction_fingerprint(
            "corine_level2", "summary", "m.gguf", 42, 0.0, ["21", "31"]
        )
        assert fp1 == fp2


class TestSourceLoading:
    def _make_temp_files(self, tmpdir):
        summaries_path = os.path.join(tmpdir, "summaries.json")
        contents_path = os.path.join(tmpdir, "contents.json")
        no_place_path = os.path.join(tmpdir, "no_place.json")
        wiki_path = os.path.join(tmpdir, "wiki.json")
        with open(summaries_path, "w") as f:
            json.dump({"100": {"summary": "Strasbourg est une ville."}}, f)
        with open(contents_path, "w") as f:
            json.dump({"100": {"content": "Full content"}}, f)
        with open(no_place_path, "w") as f:
            json.dump({"100": {"summary": "No place summary here."}}, f)
        with open(wiki_path, "w") as f:
            json.dump(
                [{"pageid": 100, "lat": 0.5, "lon": 0.5, "title": "Strasbourg"}], f
            )
        return summaries_path, contents_path, no_place_path, wiki_path

    def test_loads_summary_field_from_summary_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                _,
            ) = self._make_temp_files(tmpdir)
            from scripts.data.classify_articles import load_text_source

            result = load_text_source(
                "summary", contents_path, summaries_path, no_place_path
            )
            assert result["100"] == "Strasbourg est une ville."

    def test_loads_summary_field_from_summary_no_place_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                _,
            ) = self._make_temp_files(tmpdir)
            from scripts.data.classify_articles import load_text_source

            result = load_text_source(
                "summary_no_place", contents_path, summaries_path, no_place_path
            )
            assert result["100"] == "No place summary here."

    def test_loads_content_field_from_content_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                _,
            ) = self._make_temp_files(tmpdir)
            from scripts.data.classify_articles import load_text_source

            result = load_text_source(
                "content", contents_path, summaries_path, no_place_path
            )
            assert result["100"] == "Full content"

    def test_load_text_source_raises_on_unknown_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                _,
            ) = self._make_temp_files(tmpdir)
            from scripts.data.classify_articles import load_text_source

            with pytest.raises(ValueError, match="Unknown text source"):
                load_text_source(
                    "unknown", contents_path, summaries_path, no_place_path
                )


def _corine_temp_setup(tmpdir):
    """Returns (summaries_path, contents_path, no_place_path, wiki_path, corine_shp, output_dir)."""
    summaries_path = os.path.join(tmpdir, "summaries.json")
    contents_path = os.path.join(tmpdir, "contents.json")
    no_place_path = os.path.join(tmpdir, "no_place.json")
    wiki_path = os.path.join(tmpdir, "wiki.json")
    output_dir = os.path.join(tmpdir, "out")
    corine_shp = os.path.join(tmpdir, "corine.shp")
    os.makedirs(output_dir, exist_ok=True)
    with open(summaries_path, "w") as f:
        json.dump({"100": {"summary": "Une forêt."}}, f)
    with open(contents_path, "w") as f:
        json.dump({"100": {"content": "Full"}}, f)
    with open(no_place_path, "w") as f:
        json.dump({"100": {"summary": "No place"}}, f)
    with open(wiki_path, "w") as f:
        json.dump([{"pageid": 100, "lat": 0.5, "lon": 0.5, "title": "Forêt"}], f)
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"code_18": ["311"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
    )
    gdf.to_file(corine_shp)
    return summaries_path, contents_path, no_place_path, wiki_path, corine_shp, output_dir


def _osm_temp_setup(tmpdir):
    """Returns (summaries_path, contents_path, no_place_path, wiki_path, osm_geojson, output_dir)."""
    summaries_path = os.path.join(tmpdir, "summaries.json")
    contents_path = os.path.join(tmpdir, "contents.json")
    no_place_path = os.path.join(tmpdir, "no_place.json")
    wiki_path = os.path.join(tmpdir, "wiki.json")
    output_dir = os.path.join(tmpdir, "out")
    osm_geojson = os.path.join(tmpdir, "osm.geojson")
    os.makedirs(output_dir, exist_ok=True)
    with open(summaries_path, "w") as f:
        json.dump({"100": {"summary": "Prairie."}}, f)
    with open(contents_path, "w") as f:
        json.dump({"100": {"content": "Full"}}, f)
    with open(no_place_path, "w") as f:
        json.dump({"100": {"summary": "No place"}}, f)
    with open(wiki_path, "w") as f:
        json.dump([{"pageid": 100, "lat": 0.5, "lon": 0.5, "title": "Prairie"}], f)
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"osm_id": ["way/1"], "landuse": ["meadow"], "natural": [None]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    gdf.to_file(osm_geojson, driver="GeoJSON")
    return summaries_path, contents_path, no_place_path, wiki_path, osm_geojson, output_dir


class TestPredictionRecordShape:
    def test_record_contains_all_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                wiki_path,
                corine_shp,
                output_dir,
            ) = _corine_temp_setup(tmpdir)
            classifier = MagicMock()
            classifier.classify_single_label.return_value = {
                "prediction": "31",
                "prediction_labels": ["31"],
                "parse_status": "ok",
                "error": None,
                "raw_response": '{"label":"31"}',
                "metadata": {
                    "task": "corine_level2",
                    "text_source": "summary",
                    "model": "m.gguf",
                    "seed": 42,
                    "temperature": 0.0,
                    "prompt": "...",
                    "system_prompt": "...",
                    "allowed_labels": ["31"],
                },
            }
            from scripts.data.classify_articles import main

            with patch(
                "scripts.data.classify_articles.LLMClassifier", return_value=classifier
            ):
                main(
                    [
                        "--task",
                        "corine_level2",
                        "--text-source",
                        "summary",
                        "--wiki-articles-path",
                        wiki_path,
                        "--article-contents-path",
                        contents_path,
                        "--article-summaries-path",
                        summaries_path,
                        "--article-summaries-no-place-path",
                        no_place_path,
                        "--corine-polygons-path",
                        corine_shp,
                        "--output-dir",
                        output_dir,
                        "--model-path",
                        "m.gguf",
                    ]
                )
            pred_path = os.path.join(
                output_dir, "corine_level2_summary_predictions.json"
            )
            with open(pred_path) as f:
                records = json.load(f)
            rec = records["100"]
            for field in (
                "pageid",
                "title",
                "target",
                "prediction",
                "prediction_labels",
                "parse_status",
                "raw_response",
                "error",
                "metadata",
            ):
                assert field in rec, f"Missing field: {field}"
            assert rec["parse_status"] == "ok"
            assert rec["metadata"]["fingerprint"]
            assert "prompt" in rec["metadata"]
            assert "system_prompt" in rec["metadata"]
            assert "error" in rec


class TestResumability:
    def test_skips_existing_ok_with_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                wiki_path,
                corine_shp,
                output_dir,
            ) = _corine_temp_setup(tmpdir)
            classifier = MagicMock()
            fp = prediction_fingerprint(
                "corine_level2", "summary", "m.gguf", 42, 0.0, ["31"]
            )
            classifier.classify_single_label.return_value = {
                "prediction": "31",
                "parse_status": "ok",
                "error": None,
                "raw_response": '{"label":"31"}',
                "metadata": {
                    "task": "corine_level2",
                    "text_source": "summary",
                    "model": "m.gguf",
                    "seed": 42,
                    "temperature": 0.0,
                    "prompt": "...",
                    "system_prompt": "...",
                    "allowed_labels": ["31"],
                    "fingerprint": fp,
                },
            }
            pred_path = os.path.join(
                output_dir, "corine_level2_summary_predictions.json"
            )
            with open(pred_path, "w") as f:
                json.dump(
                    {
                        "100": {
                            "pageid": "100",
                            "title": "Forêt",
                            "target": "31",
                            "prediction": "31",
                            "parse_status": "ok",
                            "raw_response": '{"label":"31"}',
                            "error": None,
                            "metadata": {
                                "task": "corine_level2",
                                "text_source": "summary",
                                "model": "m.gguf",
                                "seed": 42,
                                "temperature": 0.0,
                                "prompt": "...",
                                "system_prompt": "...",
                                "allowed_labels": ["31"],
                                "fingerprint": fp,
                            },
                        }
                    },
                    f,
                )
            from scripts.data.classify_articles import main

            with patch(
                "scripts.data.classify_articles.LLMClassifier", return_value=classifier
            ):
                main(
                    [
                        "--task",
                        "corine_level2",
                        "--text-source",
                        "summary",
                        "--wiki-articles-path",
                        wiki_path,
                        "--article-contents-path",
                        contents_path,
                        "--article-summaries-path",
                        summaries_path,
                        "--article-summaries-no-place-path",
                        no_place_path,
                        "--corine-polygons-path",
                        corine_shp,
                        "--output-dir",
                        output_dir,
                        "--model-path",
                        "m.gguf",
                    ]
                )
            classifier.classify_single_label.assert_not_called()

    def test_overwrites_existing_error_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                wiki_path,
                corine_shp,
                output_dir,
            ) = _corine_temp_setup(tmpdir)
            classifier = MagicMock()
            fp = prediction_fingerprint(
                "corine_level2", "summary", "m.gguf", 42, 0.0, ["31"]
            )
            classifier.classify_single_label.return_value = {
                "prediction": "31",
                "parse_status": "ok",
                "error": None,
                "raw_response": '{"label":"31"}',
                "metadata": {
                    "task": "corine_level2",
                    "text_source": "summary",
                    "model": "m.gguf",
                    "seed": 42,
                    "temperature": 0.0,
                    "prompt": "...",
                    "system_prompt": "...",
                    "allowed_labels": ["31"],
                    "fingerprint": fp,
                },
            }
            pred_path = os.path.join(
                output_dir, "corine_level2_summary_predictions.json"
            )
            with open(pred_path, "w") as f:
                json.dump(
                    {
                        "100": {
                            "pageid": "100",
                            "title": "Forêt",
                            "target": "31",
                            "prediction": None,
                            "parse_status": "error",
                            "raw_response": None,
                            "error": "LLM failed",
                            "metadata": {
                                "task": "corine_level2",
                                "text_source": "summary",
                                "model": "m.gguf",
                                "seed": 42,
                                "temperature": 0.0,
                                "prompt": "...",
                                "system_prompt": "...",
                                "allowed_labels": ["31"],
                                "fingerprint": fp,
                            },
                        }
                    },
                    f,
                )
            from scripts.data.classify_articles import main

            with patch(
                "scripts.data.classify_articles.LLMClassifier", return_value=classifier
            ):
                main(
                    [
                        "--task",
                        "corine_level2",
                        "--text-source",
                        "summary",
                        "--wiki-articles-path",
                        wiki_path,
                        "--article-contents-path",
                        contents_path,
                        "--article-summaries-path",
                        summaries_path,
                        "--article-summaries-no-place-path",
                        no_place_path,
                        "--corine-polygons-path",
                        corine_shp,
                        "--output-dir",
                        output_dir,
                        "--model-path",
                        "m.gguf",
                    ]
                )
            classifier.classify_single_label.assert_called_once()


class TestOSMRunner:
    def test_osm_task_uses_multilabel_classifier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                wiki_path,
                osm_geojson,
                output_dir,
            ) = _osm_temp_setup(tmpdir)
            classifier = MagicMock()
            fp = prediction_fingerprint(
                "osm", "summary", "m.gguf", 42, 0.0, ["meadow"]
            )
            classifier.classify_multilabel.return_value = {
                "prediction": ["meadow"],
                "parse_status": "ok",
                "error": None,
                "raw_response": '{"labels":["meadow"]}',
                "metadata": {
                    "task": "osm",
                    "text_source": "summary",
                    "model": "m.gguf",
                    "seed": 42,
                    "temperature": 0.0,
                    "prompt": "...",
                    "system_prompt": "...",
                    "allowed_labels": ["meadow"],
                    "fingerprint": fp,
                },
            }
            from scripts.data.classify_articles import main

            with patch(
                "scripts.data.classify_articles.LLMClassifier", return_value=classifier
            ):
                main(
                    [
                        "--task",
                        "osm",
                        "--text-source",
                        "summary",
                        "--wiki-articles-path",
                        wiki_path,
                        "--article-contents-path",
                        contents_path,
                        "--article-summaries-path",
                        summaries_path,
                        "--article-summaries-no-place-path",
                        no_place_path,
                        "--osm-polygons-path",
                        osm_geojson,
                        "--output-dir",
                        output_dir,
                        "--model-path",
                        "m.gguf",
                    ]
                )
            classifier.classify_multilabel.assert_called_once()
            metrics_path = os.path.join(output_dir, "osm_summary_metrics.json")
            with open(metrics_path) as f:
                metrics = json.load(f)
            assert metrics["task"] == "osm"
            assert "n_eligible" in metrics
            assert "n_predicted_ok" in metrics


class TestMetricsOutput:
    def test_metrics_file_has_all_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                summaries_path,
                contents_path,
                no_place_path,
                wiki_path,
                corine_shp,
                output_dir,
            ) = _corine_temp_setup(tmpdir)
            classifier = MagicMock()
            fp = prediction_fingerprint(
                "corine_level2", "summary", "m.gguf", 42, 0.0, ["31"]
            )
            classifier.classify_single_label.return_value = {
                "prediction": "31",
                "parse_status": "ok",
                "error": None,
                "raw_response": '{"label":"31"}',
                "metadata": {
                    "task": "corine_level2",
                    "text_source": "summary",
                    "model": "m.gguf",
                    "seed": 42,
                    "temperature": 0.0,
                    "prompt": "...",
                    "system_prompt": "...",
                    "allowed_labels": ["31"],
                    "fingerprint": fp,
                },
            }
            from scripts.data.classify_articles import main

            with patch(
                "scripts.data.classify_articles.LLMClassifier", return_value=classifier
            ):
                main(
                    [
                        "--task",
                        "corine_level2",
                        "--text-source",
                        "summary",
                        "--wiki-articles-path",
                        wiki_path,
                        "--article-contents-path",
                        contents_path,
                        "--article-summaries-path",
                        summaries_path,
                        "--article-summaries-no-place-path",
                        no_place_path,
                        "--corine-polygons-path",
                        corine_shp,
                        "--output-dir",
                        output_dir,
                        "--model-path",
                        "m.gguf",
                    ]
                )
            metrics_path = os.path.join(
                output_dir, "corine_level2_summary_metrics.json"
            )
            with open(metrics_path) as f:
                metrics = json.load(f)
            for field in (
                "n_eligible",
                "n_predicted_ok",
                "n_parse_error",
                "coverage",
                "task",
                "text_source",
                "allowed_labels",
                "labels_evaluated",
            ):
                assert field in metrics, f"Missing metrics field: {field}"


class TestLimit:
    def test_limit_restricts_eligible_and_metrics_denominator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summaries_path = os.path.join(tmpdir, "summaries.json")
            contents_path = os.path.join(tmpdir, "contents.json")
            no_place_path = os.path.join(tmpdir, "no_place.json")
            wiki_path = os.path.join(tmpdir, "wiki.json")
            output_dir = os.path.join(tmpdir, "out")
            corine_shp = os.path.join(tmpdir, "corine.shp")
            os.makedirs(output_dir, exist_ok=True)
            with open(summaries_path, "w") as f:
                json.dump(
                    {
                        "100": {"summary": "Forêt A"},
                        "200": {"summary": "Forêt B"},
                        "300": {"summary": "Forêt C"},
                    },
                    f,
                )
            with open(contents_path, "w") as f:
                json.dump(
                    {
                        "100": {"content": "A"},
                        "200": {"content": "B"},
                        "300": {"content": "C"},
                    },
                    f,
                )
            with open(no_place_path, "w") as f:
                json.dump(
                    {
                        "100": {"summary": "A"},
                        "200": {"summary": "B"},
                        "300": {"summary": "C"},
                    },
                    f,
                )
            with open(wiki_path, "w") as f:
                json.dump(
                    [
                        {"pageid": 100, "lat": 0.5, "lon": 0.5, "title": "Forêt A"},
                        {"pageid": 200, "lat": 1.5, "lon": 1.5, "title": "Forêt B"},
                        {"pageid": 300, "lat": 2.5, "lon": 2.5, "title": "Forêt C"},
                    ],
                    f,
                )
            import geopandas as gpd
            from shapely.geometry import box

            gdf = gpd.GeoDataFrame(
                {"code_18": ["311", "311", "311"]},
                geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
                crs="EPSG:4326",
            )
            gdf.to_file(corine_shp)
            classifier = MagicMock()
            fp = prediction_fingerprint(
                "corine_level2", "summary", "m.gguf", 42, 0.0, ["31"]
            )
            classifier.classify_single_label.return_value = {
                "prediction": "31",
                "parse_status": "ok",
                "error": None,
                "raw_response": '{"label":"31"}',
                "metadata": {
                    "task": "corine_level2",
                    "text_source": "summary",
                    "model": "m.gguf",
                    "seed": 42,
                    "temperature": 0.0,
                    "prompt": "...",
                    "system_prompt": "...",
                    "allowed_labels": ["31"],
                    "fingerprint": fp,
                },
            }
            from scripts.data.classify_articles import main

            with patch(
                "scripts.data.classify_articles.LLMClassifier", return_value=classifier
            ):
                main(
                    [
                        "--task",
                        "corine_level2",
                        "--text-source",
                        "summary",
                        "--wiki-articles-path",
                        wiki_path,
                        "--article-contents-path",
                        contents_path,
                        "--article-summaries-path",
                        summaries_path,
                        "--article-summaries-no-place-path",
                        no_place_path,
                        "--corine-polygons-path",
                        corine_shp,
                        "--output-dir",
                        output_dir,
                        "--model-path",
                        "m.gguf",
                        "--limit",
                        "2",
                    ]
                )
            metrics_path = os.path.join(
                output_dir, "corine_level2_summary_metrics.json"
            )
            with open(metrics_path) as f:
                metrics = json.load(f)
            assert metrics["n_eligible"] == 2
