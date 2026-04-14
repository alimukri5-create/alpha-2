"""Tests for technical structure calculations."""

from __future__ import annotations

import pandas as pd

from alphaforge.models.technicals import calculate_technical_structure


def make_price_frame() -> pd.DataFrame:
    """Create an upward-trending OHLCV frame for technical tests."""
    close = [50 + (index * 0.7) for index in range(220)]
    data = {
        "Date": pd.date_range("2023-01-01", periods=len(close), freq="B"),
        "Open": [value - 0.5 for value in close],
        "High": [value + 1.5 for value in close],
        "Low": [value - 1.2 for value in close],
        "Close": close,
        "Volume": [500_000] * len(close),
    }
    return pd.DataFrame(data)


def test_technical_structure_returns_expected_fields() -> None:
    """The technical module should expose a stable output contract."""
    snapshot = calculate_technical_structure(make_price_frame()).to_dict()

    assert set(snapshot) == {
        "current_price",
        "atr_14",
        "support",
        "resistance",
        "breakout_level",
        "moving_averages",
        "trend_label",
        "structure_summary",
    }


def test_support_resistance_and_atr_are_present_with_enough_history() -> None:
    """Support, resistance, and ATR should be computed for a long enough sample."""
    snapshot = calculate_technical_structure(make_price_frame())

    assert snapshot.atr_14 is not None and snapshot.atr_14 > 0
    assert snapshot.support is not None
    assert snapshot.resistance is not None
    assert snapshot.breakout_level is not None
    assert snapshot.support < snapshot.resistance
