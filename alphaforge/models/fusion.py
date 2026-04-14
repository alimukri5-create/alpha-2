"""Rule-based fusion layer for the AlphaForge MVP.

This module combines realized volatility and technical structure into a
first-pass trade map without pretending to be a predictive model.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphaforge.models.realized_vol import RealizedVolSnapshot
from alphaforge.models.technicals import TechnicalSnapshot


@dataclass(frozen=True)
class TradeMap:
    """Phase 1 output contract for the execution map."""

    distribution_summary: str
    regime_summary: str
    starter_buy_zone: str
    size_buy_zone: str
    no_trade_zone: str
    breakout_confirmation_zone: str
    tactical_stop: str
    structural_stop: str
    confidence_summary: str
    thesis_summary: str
    driver_summary: str
    fragility_notes: list[str]

    def to_dict(self) -> dict:
        """Return a stable dictionary shape for the UI and tests."""
        return {
            "distribution_summary": self.distribution_summary,
            "regime_summary": self.regime_summary,
            "starter_buy_zone": self.starter_buy_zone,
            "size_buy_zone": self.size_buy_zone,
            "no_trade_zone": self.no_trade_zone,
            "breakout_confirmation_zone": self.breakout_confirmation_zone,
            "tactical_stop": self.tactical_stop,
            "structural_stop": self.structural_stop,
            "confidence_summary": self.confidence_summary,
            "thesis_summary": self.thesis_summary,
            "driver_summary": self.driver_summary,
            "fragility_notes": self.fragility_notes,
        }


def _format_zone(lower: float | None, upper: float | None) -> str:
    """Format a human-readable price zone."""
    if lower is None or upper is None:
        return "Not enough data yet."
    lower_bound = min(lower, upper)
    upper_bound = max(lower, upper)
    return f"${lower_bound:.2f} to ${upper_bound:.2f}"


def _pick_anchor_support(technical: TechnicalSnapshot) -> float:
    """Choose the main support anchor for buy-zone construction."""
    if technical.support is not None:
        return technical.support

    ma_20 = technical.moving_averages.get("ma_20")
    if ma_20 is not None:
        return ma_20

    return technical.current_price


def build_trade_map(
    realized_vol: RealizedVolSnapshot,
    technical: TechnicalSnapshot,
    options_summary: str | None = None,
) -> TradeMap:
    """Combine Phase 1 layers into a transparent trade map."""
    current_price = technical.current_price
    atr = technical.atr_14 or max(current_price * 0.02, 0.01)
    anchor_support = _pick_anchor_support(technical)

    starter_lower = anchor_support - (0.25 * atr)
    starter_upper = anchor_support + (0.50 * atr)
    size_lower = anchor_support - (1.00 * atr)
    size_upper = anchor_support + (0.10 * atr)
    no_trade_lower = current_price - (0.50 * atr)
    no_trade_upper = current_price + (0.50 * atr)

    breakout_reference = technical.breakout_level or (
        (technical.resistance or current_price) + (0.50 * atr)
    )
    breakout_upper = breakout_reference + (0.50 * atr)
    tactical_stop = starter_lower - (0.50 * atr)
    structural_stop = size_lower - (0.75 * atr)

    confidence_parts = []
    if technical.trend_label in {"uptrend", "range-to-up"}:
        confidence_parts.append("trend support is constructive")
    else:
        confidence_parts.append("trend support is only partial")

    if realized_vol.regime_label in {"calm", "normal", "cooling-vol"}:
        confidence_parts.append("realized volatility is manageable")
    else:
        confidence_parts.append("realized volatility is elevated")

    fragility_notes = [
        "Phase 1 has no live options-implied probability map yet.",
        "Jump and event risk layers are not implemented yet, so stops should be treated conservatively.",
    ]

    distribution_summary = (
        options_summary
        or "Options-implied distribution is not available in Phase 1, so the range view uses realized volatility as a proxy."
    )
    regime_summary = (
        f"Price structure is {technical.trend_label} while realized volatility is {realized_vol.regime_label}."
    )
    thesis_summary = (
        "This first-pass map prefers risk-taking near support and avoids chasing price in the middle of the current range."
    )
    driver_summary = (
        f"The conclusion is driven mainly by support near {anchor_support:.2f}, "
        f"ATR near {atr:.2f}, and a {realized_vol.regime_label} realized-vol backdrop."
    )

    return TradeMap(
        distribution_summary=distribution_summary,
        regime_summary=regime_summary,
        starter_buy_zone=_format_zone(starter_lower, starter_upper),
        size_buy_zone=_format_zone(size_lower, size_upper),
        no_trade_zone=_format_zone(no_trade_lower, no_trade_upper),
        breakout_confirmation_zone=_format_zone(breakout_reference, breakout_upper),
        tactical_stop=f"Below ${tactical_stop:.2f}",
        structural_stop=f"Below ${structural_stop:.2f}",
        confidence_summary="; ".join(confidence_parts).capitalize() + ".",
        thesis_summary=thesis_summary,
        driver_summary=driver_summary,
        fragility_notes=fragility_notes,
    )
