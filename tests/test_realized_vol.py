"""Tests for realized volatility calculations."""

from __future__ import annotations

import math

import pandas as pd

from alphaforge.models.realized_vol import calculate_realized_volatility


def make_price_frame() -> pd.DataFrame:
    """Create a stable synthetic daily OHLCV frame for tests."""
    close = [100 + (index * 0.8) + ((-1) ** index * 0.6) for index in range(80)]
    data = {
        "Date": pd.date_range("2024-01-01", periods=len(close), freq="B"),
        "Open": [value - 0.4 for value in close],
        "High": [value + 1.2 for value in close],
        "Low": [value - 1.1 for value in close],
        "Close": close,
        "Volume": [1_000_000] * len(close),
    }
    return pd.DataFrame(data)


def test_realized_volatility_outputs_expected_keys() -> None:
    """The realized-vol module should return the advertised output shape."""
    snapshot = calculate_realized_volatility(make_price_frame()).to_dict()

    assert set(snapshot) == {
        "hv_10",
        "hv_20",
        "hv_60",
        "current_price",
        "short_horizon_move",
        "short_horizon_move_pct",
        "regime_label",
        "summary",
    }


def test_realized_volatility_values_are_non_negative() -> None:
    """Historical-vol metrics should be finite and non-negative when enough data exists."""
    snapshot = calculate_realized_volatility(make_price_frame())

    assert snapshot.hv_10 is not None and snapshot.hv_10 >= 0
    assert snapshot.hv_20 is not None and snapshot.hv_20 >= 0
    assert snapshot.hv_60 is not None and snapshot.hv_60 >= 0
    assert snapshot.short_horizon_move is not None and snapshot.short_horizon_move >= 0
    assert not math.isnan(snapshot.current_price)
