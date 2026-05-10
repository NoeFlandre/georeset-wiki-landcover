"""Classification type aliases.

The canonical JSON-facing contracts live in :mod:`georeset.contracts`; this module
keeps the existing classification import path stable.
"""

from georeset.contracts import (
    ClassificationTarget,
    ClassificationTask,
    ParseStatus,
    PredictionRecord,
    PredictionResult,
)

__all__ = [
    "ClassificationTarget",
    "ClassificationTask",
    "ParseStatus",
    "PredictionRecord",
    "PredictionResult",
]
