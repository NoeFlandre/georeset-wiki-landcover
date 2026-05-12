"""Tests for CORINE class-count distribution summaries."""

from unittest.mock import patch

import pandas as pd

from georeset.analysis.distribution_summary import (
    class_count_summary,
    format_class_count_summary,
    main,
)


def test_class_count_summary_handles_repeated_osm_ids(tmp_path):
    """Summary should bucket polygons by class rows per OSM id."""
    csv_path = tmp_path / "distribution.csv"
    csv_path.write_text(
        "osm_id,class_label\na,311\nb,112\nb,321\nc,211\nc,212\nd,111\n",
        encoding="utf-8",
    )

    summary = class_count_summary(str(csv_path))

    expected = pd.DataFrame(
        {
            "n_classes": [1, 2],
            "n_polygons": [2, 2],
            "pct_polygons": [50.0, 50.0],
        }
    )
    pd.testing.assert_frame_equal(summary, expected)


def test_pct_polygons_is_rounded_to_two_decimals(tmp_path):
    """Percentages should be rounded to two decimals to preserve CLI output stability."""
    csv_path = tmp_path / "distribution.csv"
    csv_path.write_text(
        "osm_id,class_label\na,311\nb,112\nb,321\nc,211\n",
        encoding="utf-8",
    )

    summary = class_count_summary(str(csv_path))

    assert summary["pct_polygons"].tolist() == [66.67, 33.33]


@patch("builtins.print")
def test_format_class_count_summary_returns_string(mock_print):
    """Formatter should return a printable summary string and never print itself."""
    summary = pd.DataFrame(
        {
            "n_classes": [1, 3],
            "n_polygons": [2, 1],
            "pct_polygons": [66.67, 33.33],
        }
    )

    text = format_class_count_summary(summary)
    assert text == summary.to_string(index=False)
    assert not mock_print.called


@patch("builtins.print")
def test_main_prints_once(mock_print, tmp_path):
    """main(argv) should emit exactly one summary print."""
    csv_path = tmp_path / "distribution.csv"
    csv_path.write_text(
        "osm_id,class_label\na,311\nb,112\nb,321\n",
        encoding="utf-8",
    )

    main([str(csv_path)])

    assert mock_print.call_count == 1
    summary = format_class_count_summary(class_count_summary(str(csv_path)))
    mock_print.assert_called_once_with(summary)
