"""Helpers for aligned-vs-shuffled metric deltas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def primary_score(row: Mapping[str, Any]) -> float:
    if row["task"] == "corine_level2":
        return float(row.get("balanced_accuracy", 0.0))
    return float(row.get("jaccard", row.get("exact_match_accuracy", 0.0)))


def primary_metric_name(task: str, *, osm_metric: str = "jaccard") -> str:
    return "balanced_accuracy" if task == "corine_level2" else osm_metric


def compute_shuffled_delta_rows(
    rows: Sequence[dict[str, Any]],
    *,
    shuffled_pairs: Mapping[str, str],
    model_columns: Sequence[str],
) -> list[dict[str, Any]]:
    key_columns = ("subset", *model_columns, "task", "text_source")
    by_key = {tuple(row[column] for column in key_columns): row for row in rows}
    deltas: list[dict[str, Any]] = []

    for aligned, shuffled in shuffled_pairs.items():
        for row in rows:
            if row["text_source"] != aligned:
                continue
            shuffled_key = tuple(
                shuffled if column == "text_source" else row[column] for column in key_columns
            )
            shuffled_row = by_key.get(shuffled_key)
            if shuffled_row is None:
                continue
            aligned_score = primary_score(row)
            shuffled_score = primary_score(shuffled_row)
            deltas.append(
                {
                    "subset": row["subset"],
                    **{column: row[column] for column in model_columns},
                    "task": row["task"],
                    "text_source": aligned,
                    "shuffled_text_source": shuffled,
                    "primary_metric": primary_metric_name(str(row["task"])),
                    "aligned_score": aligned_score,
                    "shuffled_score": shuffled_score,
                    "delta": aligned_score - shuffled_score,
                    "n_aligned": row["n"],
                    "n_shuffled": shuffled_row["n"],
                }
            )
    return deltas
