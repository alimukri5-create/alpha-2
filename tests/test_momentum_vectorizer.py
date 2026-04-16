"""Tests for the manual momentum vectorizer."""

from __future__ import annotations

import json

from alphaforge.models.momentum_vectorizer import (
    MomentumSignalVectorizer,
    export_signals_json,
)


def make_signal(**overrides) -> dict:
    """Build one representative signal with easy override hooks."""
    vectorizer = MomentumSignalVectorizer()
    signal = vectorizer.vectorize_signal(
        ticker="UUUU",
        current_price=6.45,
        entry_signal_strength=7.5,
        confluence_factors=[
            "compression_breakout_forming",
            "volume_spike_2.1x_20day_avg",
            "sector_uranium_strength",
            "countertrend_to_spot_pullback",
        ],
        catalyst_type="supply_shock",
        catalyst_magnitude="major",
        catalyst_timing="this_week",
        price_action_timeframe="1d",
        vol_vs_20day_avg=2.1,
        price_momentum_5d=3.2,
        debt_to_mcap=0.18,
        swing_target_days=5,
        prior_pattern_win_rate=0.62,
        notes="Bull thesis intact.",
        **overrides,
    )
    return signal.to_dict()


def test_shariah_screen_fails_on_excluded_ticker() -> None:
    """Excluded tickers should fail even if the debt ratio is acceptable."""
    result = MomentumSignalVectorizer().shariah_screen("LMT", 0.10)

    assert result["passed"] is False
    assert "sector_exclusion:weapons" in result["violations"]


def test_shariah_screen_fails_on_debt_threshold() -> None:
    """Debt ratios at or above 33% should fail."""
    result = MomentumSignalVectorizer().shariah_screen("AAPL", 0.33)

    assert result["passed"] is False
    assert "debt_ratio_exceeds_33_percent" in result["violations"]


def test_vectorizer_produces_tier_one_for_strong_setup() -> None:
    """Strong, aligned signals should clear the Tier 1 bar."""
    signal = make_signal()

    assert signal["scores"]["composite_quality"] >= 0.75
    assert signal["local_tier"] == "Tier 1"


def test_vectorizer_produces_tier_three_for_weak_setup() -> None:
    """Weak signals should fall into Tier 3 when Shariah passes."""
    signal = make_signal(
        ticker="RKLB",
        current_price=18.34,
        entry_signal_strength=4.2,
        confluence_factors=["price_above_200ma"],
        catalyst_type="none",
        catalyst_magnitude="minor",
        catalyst_timing="uncertain",
        price_action_timeframe="weekly",
        vol_vs_20day_avg=0.95,
        price_momentum_5d=-0.8,
        debt_to_mcap=0.19,
        prior_pattern_win_rate=0.50,
        notes="Weak setup.",
    )

    assert signal["scores"]["composite_quality"] < 0.55
    assert signal["local_tier"] == "Tier 3"


def test_vectorizer_disqualifies_shariah_failures() -> None:
    """Shariah failures should zero out composite quality and disqualify the signal."""
    signal = make_signal(ticker="BAC", debt_to_mcap=0.10)

    assert signal["scores"]["composite_quality"] == 0.0
    assert signal["local_tier"] == "Disqualified"


def test_classification_boundary_rules_are_stable() -> None:
    """Tier boundaries should match the v1 contract exactly."""
    vectorizer = MomentumSignalVectorizer()

    assert vectorizer.classify_local_tier(0.75, True) == "Tier 1"
    assert vectorizer.classify_local_tier(0.74, True) == "Tier 2"
    assert vectorizer.classify_local_tier(0.55, True) == "Tier 2"
    assert vectorizer.classify_local_tier(0.54, True) == "Tier 3"
    assert vectorizer.classify_local_tier(0.80, False) == "Disqualified"


def test_export_json_includes_stable_top_level_keys() -> None:
    """JSON export should preserve the agreed signal schema."""
    payload = json.loads(export_signals_json([make_signal()]))
    signal = payload[0]

    assert set(signal) == {
        "timestamp",
        "ticker",
        "price",
        "signal_strength",
        "confluence_factors",
        "confluence_count",
        "catalyst",
        "scores",
        "local_tier",
        "shariah_compliance",
        "price_action",
        "notes",
        "metadata",
    }
