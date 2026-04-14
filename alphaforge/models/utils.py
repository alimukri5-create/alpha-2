"""Small shared helpers for AlphaForge model modules."""

from __future__ import annotations


def safe_round(value: float | None, digits: int = 2) -> float | None:
    """Round when possible while preserving missing values."""
    if value is None:
        return None
    return round(float(value), digits)
