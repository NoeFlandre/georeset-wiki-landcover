"""Tests for WikiArticleTypeFetcher."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from georeset.fetchers.wiki_article_type_fetcher import WikiArticleTypeFetcher


def _response_json(pageid_to_records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    pages = {str(pageid): payload for pageid, payload in pageid_to_records.items()}
    return {"query": {"pages": pages}}


def _make_response(json_payload: dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.headers = {"Content-Type": "application/json"}
    response.json.return_value = json_payload
    response.status_code = status_code
    response.ok = status_code < 400

    def _raise_for_status() -> None:
        if status_code >= 400:
            raise requests.HTTPError(response=response)

    response.raise_for_status.side_effect = _raise_for_status
    response.text = ""
    return response


def test_fetcher_batches_pageids(tmp_path: Path) -> None:
    fetcher = WikiArticleTypeFetcher()

    def side_effect(*_args: object, **kwargs: object) -> MagicMock:
        params = kwargs.get("params", {})
        requested = str(params.get("pageids", ""))
        pageids = [int(pid) for pid in requested.split("|") if pid]
        payload = _response_json(
            {pid: {"pageid": pid, "title": f"P{pid}", "ns": 0} for pid in pageids}
        )
        return _make_response(payload)

    with patch("georeset.fetchers.wiki_article_type_fetcher.requests.get", side_effect=side_effect) as get_mock:
        input_path = tmp_path / "wiki_articles.json"
        output_path = tmp_path / "article_type_metadata.json"
        input_path.write_text(
            json.dumps([{"pageid": 1}, {"pageid": 2}, {"pageid": 3}, {"pageid": 4}]),
            encoding="utf-8",
        )

        fetcher.fetch_from_file(input_path, output_path, batch_size=2)

    assert get_mock.call_count == 2
    requested_batches = [call.kwargs["params"]["pageids"] for call in get_mock.call_args_list]
    assert requested_batches == ["1|2", "3|4"]


def test_fetcher_resumes_current_records(tmp_path: Path) -> None:
    fetcher = WikiArticleTypeFetcher()
    payload = _response_json(
        {
            1: {"pageid": 1, "title": "T1", "ns": 0, "categories": []},
            3: {"pageid": 3, "title": "T3", "ns": 0, "categories": []},
        }
    )

    with patch("georeset.fetchers.wiki_article_type_fetcher.requests.get", return_value=_make_response(payload)) as get_mock:
        input_path = tmp_path / "wiki_articles.json"
        output_path = tmp_path / "article_type_metadata.json"
        input_path.write_text(
            json.dumps([{"pageid": 1}, {"pageid": 2}, {"pageid": 3}],
            ),
            encoding="utf-8",
        )
        output_path.write_text(
            json.dumps(
                {
                    "2": {
                        "pageid": "2",
                        "title": "Already",
                        "primary_article_type": "other_or_unclear",
                        "candidate_article_types": ["other_or_unclear"],
                        "matched_categories": [],
                        "matched_rules": [],
                        "all_categories_count": 0,
                        "has_categories": False,
                    }
                }
            ),
            encoding="utf-8",
        )

        fetcher.fetch_from_file(input_path, output_path, batch_size=2)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(data) == {"1", "2", "3"}
    pageids_requested = []
    for call in get_mock.call_args_list:
        pageids_requested.extend(str(call.kwargs["params"]["pageids"]).split("|"))
    assert "2" not in pageids_requested


def test_fetcher_handles_missing_categories(tmp_path: Path) -> None:
    fetcher = WikiArticleTypeFetcher()
    payload = _response_json({1: {"pageid": 1, "title": "NoCats", "ns": 0}})

    with patch("georeset.fetchers.wiki_article_type_fetcher.requests.get", return_value=_make_response(payload)):
        input_path = tmp_path / "wiki_articles.json"
        output_path = tmp_path / "article_type_metadata.json"
        input_path.write_text(json.dumps([{"pageid": 1}]), encoding="utf-8")

        fetcher.fetch_from_file(input_path, output_path, batch_size=1)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    row = data["1"]
    assert row["has_categories"] is False
    assert row["all_categories_count"] == 0
    assert row["primary_article_type"] == "other_or_unclear"


def test_fetcher_does_not_fetch_article_content(tmp_path: Path) -> None:
    fetcher = WikiArticleTypeFetcher()
    payload = _response_json({1: {"pageid": 1, "title": "A", "ns": 0, "categories": []}})

    with patch("georeset.fetchers.wiki_article_type_fetcher.requests.get", return_value=_make_response(payload)) as get_mock:
        input_path = tmp_path / "wiki_articles.json"
        output_path = tmp_path / "article_type_metadata.json"
        input_path.write_text(json.dumps([{"pageid": 1}]), encoding="utf-8")

        fetcher.fetch_from_file(input_path, output_path, batch_size=1)

    params = get_mock.call_args.kwargs["params"]
    assert "extracts" not in params
    assert "explaintext" not in params
    assert params["prop"] == "categories|pageprops"


def test_fetcher_writes_atomically(tmp_path: Path) -> None:
    fetcher = WikiArticleTypeFetcher()
    payload = _response_json({1: {"pageid": 1, "title": "A", "ns": 0, "categories": []}})

    with patch("georeset.fetchers.wiki_article_type_fetcher.write_json_atomic") as write_atomic_mock, patch(
        "georeset.fetchers.wiki_article_type_fetcher.requests.get",
        return_value=_make_response(payload),
    ):
        input_path = tmp_path / "wiki_articles.json"
        output_path = tmp_path / "article_type_metadata.json"
        input_path.write_text(json.dumps([{"pageid": 1}]), encoding="utf-8")

        fetcher.fetch_from_file(input_path, output_path, batch_size=1)

    assert write_atomic_mock.called


def test_fetcher_retries_transient_requests_exception() -> None:
    fetcher = WikiArticleTypeFetcher()
    payload = _response_json({1: {"pageid": 1, "title": "A", "ns": 0, "categories": []}})

    with patch("georeset.fetchers.wiki_article_type_fetcher.time.sleep"), patch(
        "georeset.fetchers.wiki_article_type_fetcher.requests.get",
        side_effect=[requests.RequestException("temporary"), _make_response(payload)],
    ) as get_mock:
        rows = fetcher._fetch_metadata_batch([1], max_attempts=2, base_backoff_seconds=0.0)

    assert get_mock.call_count == 2
    assert rows["1"]["title"] == "A"


def test_fetcher_does_not_retry_non_429_4xx() -> None:
    fetcher = WikiArticleTypeFetcher()
    payload = _make_response({1: {"pageid": 1, "title": "A", "ns": 0, "categories": []}}, status_code=400)

    with (
        patch("georeset.fetchers.wiki_article_type_fetcher.time.sleep"),
        patch("georeset.fetchers.wiki_article_type_fetcher.requests.get", return_value=payload) as get_mock,
        pytest.raises(requests.HTTPError),
    ):
        fetcher._fetch_metadata_batch([1], max_attempts=3, base_backoff_seconds=0.0)

    assert get_mock.call_count == 1
