"""Shared label-universe helpers for experiment metrics."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from georeset_wiki_landcover.classification.labels import CORINE_LEVEL2_DESCRIPTIONS


def label_universe(
    records: pd.DataFrame,
    task: str,
    *,
    columns: Sequence[str] = ("target", "prediction"),
) -> list[str]:
    if task == "corine_level2":
        return sorted(CORINE_LEVEL2_DESCRIPTIONS)

    labels: set[str] = set()
    for column in columns:
        if column not in records.columns:
            continue
        for values in records[column]:
            if isinstance(values, list):
                labels.update(str(value) for value in values)
            elif values is not None:
                labels.add(str(values))
    return sorted(labels)
