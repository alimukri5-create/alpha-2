"""Tests for the autonomous momentum vectorizer."""

from __future__ import annotations

import pandas as pd

from alphaforge.models.autonomous_vectorizer import (
    CatalystResult,
    _evaluate_bbw,
    _evaluate_regime,
    _evaluate_significance,
    _evaluate_volume,
    analyze_ticker,
    analyze_tickers,
    parse_ticker_csv,
    parse_ticker_text,
)


def make_daily_frame(days: int = 260, volume_scale: float = 1.0) -> pd.DataFrame:
    """Build a deterministic uptrend with an end-of-series acceleration."""
    close = [50 + (index * 0.2) for index in range(days - 5)]
    close.extend([105, 108, 112, 117, 123])
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=days, freq="B", tz="UTC"),
            "Open": [value - 1 for value in close],
            "High": [value + 2 for value in close],
            "Low": [value - 2 for value in close],
            "Close": close,
            "Volume": [1_000_000 * volume_scale] * (days - 5) + [2_000_000, 2_200_000, 2_400_000, 2_700_000, 4_000_000],
        }
    )


def make_intraday_frame(hours: int = 800) -> pd.DataFrame:
    """Build a deterministic intraday frame with enough depth for SMA checks."""
    close = [20 + (index * 0.03) for index in range(hours)]
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2025-01-01", periods=hours, freq="H", tz="UTC"),
            "Open": [value - 0.2 for value in close],
            "High": [value + 0.5 for value in close],
            "Low": [value - 0.4 for value in close],
            "Close": close,
            "Volume": [100_000] * hours,
        }
    )


def make_stock_data(debt: float = 10_000_000, market_cap: float = 100_000_000) -> dict:
    """Build a stock-data payload matching the autonomous analyzer contract."""
    return {
        "sector": "Technology",
        "industry": "Software",
        "market_cap": market_cap,
        "total_debt": debt,
        "sec_income_data": {
            "message": "SEC filing data fetched successfully.",
            "selected_non_core_income_fact": None,
            "revenue_fact": None,
        },
    }


def test_parse_helpers_accept_text_and_csv() -> None:
    """Batch helpers should normalize pasted and CSV-uploaded symbols."""
    assert parse_ticker_text("uuuu, axti\nrklb") == ["UUUU", "AXTI", "RKLB"]
    csv_payload = b"ticker\nuuuu\naxti\n"
    assert parse_ticker_csv(csv_payload) == ["UUUU", "AXTI"]


def test_bbw_detects_compression_on_flat_series() -> None:
    """A flat close series should read as compressed rather than expanded."""
    frame = make_daily_frame()
    frame.loc[220:, "Close"] = 100.0
    score, status, metrics = _evaluate_bbw(frame)

    assert status == "compressed"
    assert score >= 0.55
    assert metrics["bbw_percentile"] is not None


def test_volume_score_rewards_large_volume_spike() -> None:
    """A major volume spike plus OBV accumulation should score strongly."""
    score, metrics = _evaluate_volume(make_daily_frame())

    assert score >= 0.8
    assert metrics["obv_accumulating"] is True


def test_regime_and_significance_detect_strong_uptrend() -> None:
    """The regime and significance layers should recognize a strong trend."""
    frame = make_daily_frame()
    regime, _ = _evaluate_regime(frame)
    significance_score, significance_bonus, metrics = _evaluate_significance(frame)

    assert regime == "uptrend"
    assert significance_score in {0.35, 1.0}
    assert significance_bonus in {-0.05, 0.10}
    assert "significant" in metrics


def test_analyze_ticker_returns_tradeable_result_for_strong_setup() -> None:
    """A complete mocked setup should produce a ranked autonomous result."""
    result = analyze_ticker(
        "UUUU",
        stock_data_fetcher=lambda _: make_stock_data(),
        daily_history_fetcher=lambda _: make_daily_frame(),
        intraday_history_fetcher=lambda _: make_intraday_frame(),
        spy_history_fetcher=lambda: make_daily_frame(volume_scale=0.8),
        catalyst_fetcher=lambda _: CatalystResult(0.8, "earnings", "Earnings are scheduled this week.", "this_week", "test"),
    )

    assert result.status == "ok"
    assert result.ticker == "UUUU"
    assert result.tier in {"Tier 1", "Tier 2", "Tier 3"}
    assert result.trade_signal in {"BUY", "WATCH", "SKIP"}
    assert result.shariah.status == "PASS"


def test_analyze_ticker_disqualifies_known_shariah_failures() -> None:
    """A debt ratio above the hard limit should zero the composite."""
    result = analyze_ticker(
        "FAIL",
        stock_data_fetcher=lambda _: make_stock_data(debt=40_000_000, market_cap=100_000_000),
        daily_history_fetcher=lambda _: make_daily_frame(),
        intraday_history_fetcher=lambda _: make_intraday_frame(),
        spy_history_fetcher=lambda: make_daily_frame(),
        catalyst_fetcher=lambda _: CatalystResult(0.8, "earnings", "Earnings are scheduled this week.", "this_week", "test"),
    )

    assert result.shariah.status == "FAIL"
    assert result.scores.composite == 0.0
    assert result.trade_signal == "DISQUALIFIED"


def test_analyze_ticker_flags_missing_income_without_disqualifying() -> None:
    """Missing income facts should create warnings rather than a hard fail."""
    result = analyze_ticker(
        "FLAG",
        stock_data_fetcher=lambda _: make_stock_data(),
        daily_history_fetcher=lambda _: make_daily_frame(),
        intraday_history_fetcher=lambda _: make_intraday_frame(),
        spy_history_fetcher=lambda: make_daily_frame(),
        catalyst_fetcher=lambda _: CatalystResult(0.3, "none", "No clear catalyst detected.", "none", "test"),
    )

    assert result.shariah.status == "PASS"
    assert any("Income screening" in warning for warning in result.shariah.warnings)


def test_batch_analysis_sorts_by_tier_then_score() -> None:
    """Batch orchestration should preserve ranked output."""
    strong = analyze_ticker(
        "STRONG",
        stock_data_fetcher=lambda _: make_stock_data(),
        daily_history_fetcher=lambda _: make_daily_frame(),
        intraday_history_fetcher=lambda _: make_intraday_frame(),
        spy_history_fetcher=lambda: make_daily_frame(volume_scale=0.8),
        catalyst_fetcher=lambda _: CatalystResult(0.8, "earnings", "Earnings are scheduled this week.", "this_week", "test"),
    )
    weak = analyze_ticker(
        "WEAK",
        stock_data_fetcher=lambda _: make_stock_data(),
        daily_history_fetcher=lambda _: make_daily_frame(volume_scale=0.5),
        intraday_history_fetcher=lambda _: pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]),
        spy_history_fetcher=lambda: make_daily_frame(),
        catalyst_fetcher=lambda _: CatalystResult(0.3, "none", "No clear catalyst detected.", "none", "test"),
    )

    results = analyze_tickers(["WEAK", "STRONG"], analyzer=lambda ticker: strong if ticker == "STRONG" else weak)

    assert results[0].scores.composite >= results[-1].scores.composite
