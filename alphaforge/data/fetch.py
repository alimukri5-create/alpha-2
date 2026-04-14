"""Price-history fetch helpers for AlphaForge.

This module only handles retrieval and validation.
Analytics live in separate model modules so future data providers can be
swapped in with minimal downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class PriceFetchResult:
    """Container for fetched price history and related notes."""

    ticker: str
    period: str
    prices: pd.DataFrame | None
    notes: list[str]
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return True when price data is available."""
        return self.prices is not None and self.error is None


def _normalize_price_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Return a clean price frame with a predictable column set."""
    clean_history = history.copy()
    if isinstance(clean_history.columns, pd.MultiIndex):
        clean_history.columns = clean_history.columns.get_level_values(0)

    clean_history = clean_history.reset_index()
    if "Date" not in clean_history.columns and "Datetime" in clean_history.columns:
        clean_history = clean_history.rename(columns={"Datetime": "Date"})

    expected_columns = ["Date", *REQUIRED_PRICE_COLUMNS]
    available_columns = [column for column in expected_columns if column in clean_history.columns]
    clean_history = clean_history[available_columns].copy()

    clean_history["Date"] = pd.to_datetime(clean_history["Date"]).dt.tz_localize(None)
    clean_history = clean_history.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return clean_history


def get_price_history(ticker: str, period: str = "1y") -> PriceFetchResult:
    """Fetch daily price history for one ticker.

    The function returns a structured result instead of raising exceptions so
    the Streamlit layer can show friendly messages.
    """
    normalized_ticker = (ticker or "").strip().upper()
    notes = [
        "Price data source: Yahoo Finance via yfinance.",
        "Phase 1 uses daily price history only. Options-implied data is not implemented yet.",
    ]

    if not normalized_ticker:
        return PriceFetchResult(
            ticker=normalized_ticker,
            period=period,
            prices=None,
            notes=notes,
            error="Please enter a ticker symbol.",
        )

    try:
        history = yf.download(
            normalized_ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as error:
        return PriceFetchResult(
            ticker=normalized_ticker,
            period=period,
            prices=None,
            notes=notes,
            error=f"Could not fetch price history for {normalized_ticker}: {error}",
        )

    if history is None or history.empty:
        return PriceFetchResult(
            ticker=normalized_ticker,
            period=period,
            prices=None,
            notes=notes,
            error=f"No price history was returned for {normalized_ticker}.",
        )

    clean_history = _normalize_price_frame(history)
    missing_columns = [column for column in REQUIRED_PRICE_COLUMNS if column not in clean_history.columns]
    if missing_columns:
        return PriceFetchResult(
            ticker=normalized_ticker,
            period=period,
            prices=None,
            notes=notes,
            error=(
                f"Price history for {normalized_ticker} is missing required columns: "
                f"{', '.join(missing_columns)}."
            ),
        )

    if len(clean_history) < 30:
        notes.append("Less than 30 trading days were available, so metrics may be unstable.")

    return PriceFetchResult(
        ticker=normalized_ticker,
        period=period,
        prices=clean_history,
        notes=notes,
    )
