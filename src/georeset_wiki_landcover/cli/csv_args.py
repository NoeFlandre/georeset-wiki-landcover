"""Helpers for comma-separated CLI argument values."""

from __future__ import annotations


def parse_csv_strings(value: str) -> list[str]:
    """Parse a comma-separated string list, ignoring empty fields."""
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_csv_ints(value: str) -> list[int]:
    """Parse a comma-separated integer list, ignoring empty fields."""
    return [int(part) for part in parse_csv_strings(value)]
