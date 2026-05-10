"""Classification type aliases.

The canonical JSON-facing contracts live in :mod:`src.contracts`; this module
keeps the existing classification import path stable.
"""

from src.contracts import (
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
