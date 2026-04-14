"""Realized-volatility calculations for the AlphaForge MVP.

This module intentionally uses simple historical-volatility windows and a
rule-based regime label. The interface is structured so a future HAR-RV or
similar model can be added without breaking the UI contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from alphaforge.config import TRADING_DAYS_PER_YEAR
from alphaforge.models.utils import safe_round


@dataclass(frozen=True)
class RealizedVolSnapshot:
    """Summary of recent realized-volatility state."""

    hv_10: float | None
    hv_20: float | None
    hv_60: float | None
    current_price: float
    short_horizon_move: float | None
    short_horizon_move_pct: float | None
    regime_label: str
    summary: str

    def to_dict(self) -> dict:
        """Return a stable dictionary shape for the UI and tests."""
        return {
            "hv_10": self.hv_10,
            "hv_20": self.hv_20,
            "hv_60": self.hv_60,
            "current_price": self.current_price,
            "short_horizon_move": self.short_horizon_move,
            "short_horizon_move_pct": self.short_horizon_move_pct,
            "regime_label": self.regime_label,
            "summary": self.summary,
        }


def _compute_log_returns(close: pd.Series) -> pd.Series:
    """Calculate log returns from a close-price series."""
    price_ratio = close / close.shift(1)
    valid_ratio = price_ratio.where(price_ratio > 0)
    return valid_ratio.apply(
        lambda value: math.log(value) if pd.notna(value) else float("nan")
    ).astype("float64")


def _annualized_volatility(log_returns: pd.Series, window: int) -> float | None:
    """Calculate annualized historical volatility for a trailing window."""
    clean_returns = log_returns.dropna()
    if len(clean_returns) < window:
        return None

    rolling_std = clean_returns.tail(window).std(ddof=1)
    if pd.isna(rolling_std):
        return None
    return float(rolling_std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _label_realized_vol_regime(hv_10: float | None, hv_20: float | None, hv_60: float | None) -> str:
    """Create an honest rule-based regime label from realized-vol windows."""
    if hv_20 is None:
        return "insufficient-history"

    if hv_60 is None:
        if hv_20 < 0.20:
            return "calm"
        if hv_20 < 0.35:
            return "normal"
        return "high-vol"

    if hv_10 is not None and hv_10 > hv_20 * 1.2 and hv_20 > hv_60 * 1.1:
        return "expanding-vol"
    if hv_10 is not None and hv_10 < hv_20 * 0.85 and hv_20 < hv_60 * 0.95:
        return "cooling-vol"
    if hv_20 < 0.20:
        return "calm"
    if hv_20 < 0.35:
        return "normal"
    return "high-vol"


def calculate_realized_volatility(prices: pd.DataFrame) -> RealizedVolSnapshot:
    """Compute realized-vol summary from a daily OHLCV price frame."""
    if prices.empty:
        raise ValueError("Price history is empty.")

    close = prices["Close"].astype(float)
    current_price = float(close.iloc[-1])
    log_returns = _compute_log_returns(close)

    hv_10 = _annualized_volatility(log_returns, window=10)
    hv_20 = _annualized_volatility(log_returns, window=20)
    hv_60 = _annualized_volatility(log_returns, window=60)

    short_horizon_move_pct = hv_20 / math.sqrt(TRADING_DAYS_PER_YEAR) if hv_20 is not None else None
    short_horizon_move = current_price * short_horizon_move_pct if short_horizon_move_pct is not None else None
    regime_label = _label_realized_vol_regime(hv_10, hv_20, hv_60)

    hv_20_text = f"{hv_20:.1%}" if hv_20 is not None else "not available"
    summary = f"Realized volatility is in a {regime_label} regime. 20-day HV is {hv_20_text}."

    return RealizedVolSnapshot(
        hv_10=safe_round(hv_10, 4),
        hv_20=safe_round(hv_20, 4),
        hv_60=safe_round(hv_60, 4),
        current_price=safe_round(current_price, 2) or current_price,
        short_horizon_move=safe_round(short_horizon_move, 2),
        short_horizon_move_pct=safe_round(short_horizon_move_pct, 4),
        regime_label=regime_label,
        summary=summary,
    )
