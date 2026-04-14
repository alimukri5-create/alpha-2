"""Technical structure calculations for the AlphaForge MVP."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alphaforge.config import ATR_WINDOW, BREAKOUT_BUFFER, RESISTANCE_LOOKBACK, SUPPORT_LOOKBACK
from alphaforge.models.utils import safe_round


@dataclass(frozen=True)
class TechnicalSnapshot:
    """Structural price summary for execution planning."""

    current_price: float
    atr_14: float | None
    support: float | None
    resistance: float | None
    breakout_level: float | None
    moving_averages: dict[str, float | None]
    trend_label: str
    structure_summary: str

    def to_dict(self) -> dict:
        """Return a stable dictionary shape for the UI and tests."""
        return {
            "current_price": self.current_price,
            "atr_14": self.atr_14,
            "support": self.support,
            "resistance": self.resistance,
            "breakout_level": self.breakout_level,
            "moving_averages": self.moving_averages,
            "trend_label": self.trend_label,
            "structure_summary": self.structure_summary,
        }


def _moving_average(close: pd.Series, window: int) -> float | None:
    """Return the most recent simple moving average."""
    if len(close) < window:
        return None
    value = close.rolling(window=window).mean().iloc[-1]
    return None if pd.isna(value) else float(value)


def _average_true_range(prices: pd.DataFrame, window: int = ATR_WINDOW) -> float | None:
    """Calculate classic ATR from daily OHLC data."""
    if len(prices) < window + 1:
        return None

    high = prices["High"].astype(float)
    low = prices["Low"].astype(float)
    close = prices["Close"].astype(float)
    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(window=window).mean().iloc[-1]
    return None if pd.isna(atr) else float(atr)


def _trend_label(current_price: float, moving_averages: dict[str, float | None]) -> str:
    """Create a transparent rule-based trend label."""
    ma_20 = moving_averages["ma_20"]
    ma_50 = moving_averages["ma_50"]
    ma_200 = moving_averages["ma_200"]

    if ma_20 and ma_50 and ma_200 and current_price > ma_20 > ma_50 > ma_200:
        return "uptrend"
    if ma_20 and ma_50 and ma_200 and current_price < ma_20 < ma_50 < ma_200:
        return "downtrend"
    if ma_50 and ma_200 and current_price > ma_200 and ma_50 > ma_200:
        return "range-to-up"
    if ma_50 and ma_200 and current_price < ma_200 and ma_50 < ma_200:
        return "range-to-down"
    return "mixed"


def _format_optional_price(value: float | None) -> str:
    """Format optional price levels for plain-English summaries."""
    if value is None:
        return "not available"
    return f"{value:.2f}"


def calculate_technical_structure(prices: pd.DataFrame) -> TechnicalSnapshot:
    """Compute Phase 1 support, resistance, ATR, and trend structure."""
    if prices.empty:
        raise ValueError("Price history is empty.")

    close = prices["Close"].astype(float)
    current_price = float(close.iloc[-1])
    moving_averages = {
        "ma_20": safe_round(_moving_average(close, 20), 2),
        "ma_50": safe_round(_moving_average(close, 50), 2),
        "ma_100": safe_round(_moving_average(close, 100), 2),
        "ma_200": safe_round(_moving_average(close, 200), 2),
    }

    support = None
    resistance = None
    if len(prices) >= SUPPORT_LOOKBACK:
        support = float(prices["Low"].astype(float).tail(SUPPORT_LOOKBACK).min())
    if len(prices) >= RESISTANCE_LOOKBACK:
        resistance = float(prices["High"].astype(float).tail(RESISTANCE_LOOKBACK).max())

    atr_14 = _average_true_range(prices, ATR_WINDOW)
    breakout_level = None
    if resistance is not None and atr_14 is not None:
        breakout_level = resistance + (atr_14 * BREAKOUT_BUFFER)

    trend_label = _trend_label(current_price, moving_averages)
    structure_summary = (
        f"Trend is {trend_label}. "
        f"Price is {current_price:.2f}, support is {_format_optional_price(support)}, "
        f"and resistance is {_format_optional_price(resistance)}."
    )

    if support is None or resistance is None:
        structure_summary = (
            f"Trend is {trend_label}. More history is needed for stable support and resistance levels."
        )

    return TechnicalSnapshot(
        current_price=safe_round(current_price, 2) or current_price,
        atr_14=safe_round(atr_14, 2),
        support=safe_round(support, 2),
        resistance=safe_round(resistance, 2),
        breakout_level=safe_round(breakout_level, 2),
        moving_averages=moving_averages,
        trend_label=trend_label,
        structure_summary=structure_summary,
    )
