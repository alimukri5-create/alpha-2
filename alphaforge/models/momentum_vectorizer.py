"""Manual momentum signal vectorizer for the AlphaForge app."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime


DEFAULT_SHARIAH_EXCLUSIONS = {
    "alcohol": ["BUD", "STZ", "KO"],
    "gambling": ["LVS", "MGM", "PENN"],
    "weapons": ["RTX", "LMT", "NOC"],
    "tobacco": ["MO", "PM"],
    "financial_services": ["BAC", "GS", "MS"],
}


@dataclass(frozen=True)
class SignalVector:
    """Stable user-facing signal vector shape."""

    timestamp: str
    ticker: str
    price: float
    signal_strength: float
    confluence_factors: list[str]
    confluence_count: int
    catalyst: dict
    scores: dict
    local_tier: str
    shariah_compliance: dict
    price_action: dict
    notes: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a stable dictionary shape for JSON export and UI display."""
        return {
            "timestamp": self.timestamp,
            "ticker": self.ticker,
            "price": self.price,
            "signal_strength": self.signal_strength,
            "confluence_factors": self.confluence_factors,
            "confluence_count": self.confluence_count,
            "catalyst": self.catalyst,
            "scores": self.scores,
            "local_tier": self.local_tier,
            "shariah_compliance": self.shariah_compliance,
            "price_action": self.price_action,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class MomentumSignalVectorizer:
    """Convert manual trade observations into a structured signal vector."""

    def __init__(self, shariah_exclusions: dict[str, list[str]] | None = None):
        self.shariah_exclusions = shariah_exclusions or DEFAULT_SHARIAH_EXCLUSIONS

    def shariah_screen(self, ticker: str, debt_to_mcap: float) -> dict:
        """Run a local ticker exclusion and debt-ratio screen."""
        violations: list[str] = []
        normalized_ticker = ticker.upper()

        for category, tickers in self.shariah_exclusions.items():
            if normalized_ticker in tickers:
                violations.append(f"sector_exclusion:{category}")

        debt_ratio = round(debt_to_mcap, 3)
        if debt_ratio >= 0.33:
            violations.append("debt_ratio_exceeds_33_percent")

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "debt_to_mcap": debt_ratio,
        }

    @staticmethod
    def score_conviction(
        entry_signal_strength: float,
        confluence_count: int,
        prior_win_rate: float = 0.55,
    ) -> float:
        """Score setup conviction from signal quality, confluence, and prior edge."""
        signal_norm = min(entry_signal_strength / 10.0, 1.0)
        confluence_bonus = min(confluence_count * 0.1, 0.4)
        wr_norm = max(0.2, (prior_win_rate - 0.35) / 0.30)
        wr_norm = min(wr_norm, 1.0)
        conviction = (signal_norm * 0.5) + (confluence_bonus * 0.3) + (wr_norm * 0.2)
        return round(conviction, 2)

    @staticmethod
    def score_timeframe_alignment(
        swing_target_days: int,
        catalyst_timing: str,
        price_action_timeframe: str,
    ) -> float:
        """Score whether catalyst timing matches the intended holding period."""
        catalyst_scores = {
            "today": 1.0,
            "this_week": 0.8,
            "next_week": 0.5,
            "uncertain": 0.3,
        }
        pa_scores = {
            "intraday": 0.4,
            "1d": 0.9,
            "multiday": 0.8,
            "weekly": 0.6,
        }
        catalyst_score = catalyst_scores.get(catalyst_timing, 0.3)
        pa_score = pa_scores.get(price_action_timeframe, 0.5)
        holding_period_bonus = 0.1 if 3 <= swing_target_days <= 7 else 0.0
        alignment = (catalyst_score * 0.5) + (pa_score * 0.5) + holding_period_bonus
        return round(min(alignment, 1.0), 2)

    @staticmethod
    def score_volume_confirmation(vol_vs_20day_avg: float, price_momentum: float) -> float:
        """Score whether volume confirms the current move."""
        if vol_vs_20day_avg >= 2.0:
            vol_score = 1.0
        elif vol_vs_20day_avg >= 1.5:
            vol_score = 0.8
        elif vol_vs_20day_avg >= 1.2:
            vol_score = 0.6
        elif vol_vs_20day_avg >= 1.0:
            vol_score = 0.4
        else:
            vol_score = 0.2

        momentum_bonus = 0.1 if abs(price_momentum) >= 5.0 and vol_vs_20day_avg >= 1.5 else 0.0
        confirmation = vol_score + momentum_bonus
        return round(min(confirmation, 1.0), 2)

    @staticmethod
    def score_catalyst_strength(catalyst_type: str, catalyst_magnitude: str) -> float:
        """Score the clarity and force of the stated catalyst."""
        type_scores = {
            "earnings": 0.9,
            "macro": 0.8,
            "sector": 0.7,
            "supply_shock": 0.85,
            "insider_activity": 0.75,
            "none": 0.3,
        }
        magnitude_scores = {
            "major": 1.0,
            "moderate": 0.7,
            "minor": 0.4,
        }
        type_score = type_scores.get(catalyst_type, 0.3)
        mag_score = magnitude_scores.get(catalyst_magnitude, 0.4)
        strength = (type_score * 0.6) + (mag_score * 0.4)
        return round(strength, 2)

    @staticmethod
    def classify_local_tier(composite_quality: float, shariah_passed: bool) -> str:
        """Assign a local tier without any external LLM."""
        if not shariah_passed or composite_quality == 0:
            return "Disqualified"
        if composite_quality >= 0.75:
            return "Tier 1"
        if composite_quality >= 0.55:
            return "Tier 2"
        return "Tier 3"

    def vectorize_signal(
        self,
        *,
        ticker: str,
        current_price: float,
        entry_signal_strength: float,
        confluence_factors: list[str],
        catalyst_type: str,
        catalyst_magnitude: str,
        catalyst_timing: str,
        price_action_timeframe: str,
        vol_vs_20day_avg: float,
        price_momentum_5d: float,
        debt_to_mcap: float,
        swing_target_days: int = 5,
        prior_pattern_win_rate: float = 0.55,
        notes: str = "",
        shariah_lookup: dict | None = None,
    ) -> SignalVector:
        """Build the full structured vector for one signal."""
        conviction_score = self.score_conviction(
            entry_signal_strength, len(confluence_factors), prior_pattern_win_rate
        )
        timeframe_score = self.score_timeframe_alignment(
            swing_target_days, catalyst_timing, price_action_timeframe
        )
        volume_score = self.score_volume_confirmation(vol_vs_20day_avg, price_momentum_5d)
        catalyst_score = self.score_catalyst_strength(catalyst_type, catalyst_magnitude)
        shariah_check = self.shariah_screen(ticker, debt_to_mcap)

        composite_score = (
            conviction_score * 0.35
            + timeframe_score * 0.25
            + catalyst_score * 0.25
            + volume_score * 0.15
        )
        if not shariah_check["passed"]:
            composite_score = 0.0

        composite_score = round(composite_score, 2)
        local_tier = self.classify_local_tier(composite_score, shariah_check["passed"])

        metadata = {}
        if shariah_lookup:
            metadata["shariah_lookup"] = shariah_lookup

        return SignalVector(
            timestamp=datetime.now().isoformat(),
            ticker=ticker.upper(),
            price=round(current_price, 2),
            signal_strength=entry_signal_strength,
            confluence_factors=confluence_factors,
            confluence_count=len(confluence_factors),
            catalyst={
                "type": catalyst_type,
                "magnitude": catalyst_magnitude,
                "timing": catalyst_timing,
            },
            scores={
                "conviction": conviction_score,
                "timeframe_alignment": timeframe_score,
                "volume_confirmation": volume_score,
                "catalyst_strength": catalyst_score,
                "composite_quality": composite_score,
            },
            local_tier=local_tier,
            shariah_compliance=shariah_check,
            price_action={
                "timeframe": price_action_timeframe,
                "momentum_5d_pct": round(price_momentum_5d, 2),
                "volume_vs_20day": round(vol_vs_20day_avg, 2),
            },
            notes=notes,
            metadata=metadata,
        )


def export_signals_json(signals: list[dict]) -> str:
    """Serialize signal dictionaries into readable JSON."""
    return json.dumps(signals, indent=2)


def export_signals_for_claude(signals: list[dict]) -> str:
    """Create a copy-ready prompt for qualitative review in Claude."""
    signals_json = export_signals_json(signals)
    return f"""I've vectorized today's momentum scanner signals. Please review each signal using the local tier as a starting point, then explain whether the qualitative edge supports that tier or should be treated more cautiously.

Local tier rules:
- Tier 1: composite score >= 0.75 and Shariah compliant
- Tier 2: composite score 0.55-0.74 and Shariah compliant
- Tier 3: composite score < 0.55 and Shariah compliant
- Disqualified: failed Shariah check or composite = 0

For each signal:
1. State whether the local tier looks reasonable.
2. Explain the likely edge, fragility, and timing risks.
3. Note whether the catalyst, volume, and timeframe alignment support acting now.

Signal vectors:
{signals_json}
"""
