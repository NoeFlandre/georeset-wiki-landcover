import pandas as pd

from georeset_wiki_landcover.analysis.label_universe import label_universe
from georeset_wiki_landcover.classification.labels import CORINE_LEVEL2_DESCRIPTIONS


def test_label_universe_uses_full_allowed_corine_labels() -> None:
    records = pd.DataFrame(
        [
            {"target": "31", "prediction": "31"},
            {"target": "21", "prediction": "21"},
        ]
    )

    assert label_universe(records, "corine_level2") == sorted(CORINE_LEVEL2_DESCRIPTIONS)


def test_label_universe_collects_multilabel_values_from_target_and_prediction() -> None:
    records = pd.DataFrame(
        [
            {"target": ["wood"], "prediction": ["meadow", "wood"]},
            {"target": ["water"], "prediction": None},
        ]
    )

    assert label_universe(records, "osm") == ["meadow", "water", "wood"]


def test_label_universe_can_collect_from_target_only() -> None:
    records = pd.DataFrame(
        [
            {"target": ["wood"], "prediction": ["meadow"]},
            {"target": "water", "prediction": ["forest"]},
        ]
    )

    assert label_universe(records, "osm", columns=("target",)) == ["water", "wood"]
