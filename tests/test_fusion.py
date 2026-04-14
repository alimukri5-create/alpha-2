"""Tests for the Phase 1 fusion layer."""

from __future__ import annotations

from alphaforge.models.fusion import build_trade_map
from alphaforge.models.realized_vol import RealizedVolSnapshot
from alphaforge.models.technicals import TechnicalSnapshot


def test_trade_map_contains_required_phase_one_fields() -> None:
    """Fusion output should match the user-facing Phase 1 contract."""
    realized_vol = RealizedVolSnapshot(
        hv_10=0.22,
        hv_20=0.24,
        hv_60=0.28,
        current_price=100.0,
        short_horizon_move=1.51,
        short_horizon_move_pct=0.0151,
        regime_label="normal",
        summary="Realized volatility is in a normal regime.",
    )
    technical = TechnicalSnapshot(
        current_price=100.0,
        atr_14=2.5,
        support=97.0,
        resistance=104.0,
        breakout_level=105.25,
        moving_averages={"ma_20": 99.0, "ma_50": 96.0, "ma_100": 92.0, "ma_200": 88.0},
        trend_label="uptrend",
        structure_summary="Trend is uptrend.",
    )

    trade_map = build_trade_map(realized_vol, technical).to_dict()

    assert set(trade_map) == {
        "distribution_summary",
        "regime_summary",
        "starter_buy_zone",
        "size_buy_zone",
        "no_trade_zone",
        "breakout_confirmation_zone",
        "tactical_stop",
        "structural_stop",
        "confidence_summary",
        "thesis_summary",
        "driver_summary",
        "fragility_notes",
    }
    assert "Phase 1" in trade_map["distribution_summary"]
    assert isinstance(trade_map["fragility_notes"], list)
