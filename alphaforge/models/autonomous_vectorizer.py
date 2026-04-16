"""Autonomous momentum vectorizer and analysis engine."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import StringIO
from math import isnan
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from data_fetcher import get_stock_data


PKT = ZoneInfo("Asia/Karachi")
SECTOR_EXCLUSION_KEYWORDS = {
    "alcohol",
    "gambling",
    "weapons",
    "weapon",
    "tobacco",
    "financial services",
    "bank",
    "banks",
    "insurance",
    "credit services",
    "mortgage",
    "lending",
}


@dataclass(frozen=True)
class ScoreBreakdown:
    momentum: float
    volume: float
    confluence: float
    bbw: float
    catalyst: float
    significance: float
    regime_bonus: float
    significance_bonus: float
    composite: float
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ShariahScreenResult:
    passed: bool
    status: str
    violations: list[str]
    warnings: list[str]
    debt_to_market_cap: float | None
    sector: str | None
    industry: str | None
    income_screen_note: str


@dataclass(frozen=True)
class CatalystResult:
    score: float
    catalyst_type: str
    summary: str
    timing: str
    source: str


@dataclass(frozen=True)
class AutonomousAnalysisResult:
    ticker: str
    analyzed_at_pk_time: str
    status: str
    error: str | None
    latest_price: float | None
    price_change_5d_pct: float | None
    relative_strength_vs_spy: float | None
    volatility_regime: str
    compression_status: str
    regime: str
    market_structure: str
    tier: str
    trade_signal: str
    shariah: ShariahScreenResult
    catalyst: CatalystResult
    scores: ScoreBreakdown
    technical_metrics: dict[str, Any]
    narratives: dict[str, str]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_pkt_iso() -> str:
    return datetime.now(PKT).isoformat()


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(min(max(value, minimum), maximum))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return None if isnan(result) else result
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if isnan(result) else result


def _normalize_price_frame(history: pd.DataFrame, is_intraday: bool) -> pd.DataFrame:
    clean = history.copy()
    if isinstance(clean.columns, pd.MultiIndex):
        clean.columns = clean.columns.get_level_values(0)
    clean = clean.reset_index()
    date_column = "Datetime" if is_intraday else "Date"
    fallback = "Date" if is_intraday else "Datetime"
    if date_column not in clean.columns and fallback in clean.columns:
        clean = clean.rename(columns={fallback: date_column})
    columns = [date_column, "Open", "High", "Low", "Close", "Volume"]
    clean = clean[[column for column in columns if column in clean.columns]].copy()
    clean[date_column] = pd.to_datetime(clean[date_column], utc=True, errors="coerce")
    return clean.dropna(subset=[date_column, "Close"]).sort_values(date_column).reset_index(drop=True)


def _download_history(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        history = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if history is None or history.empty:
        return None
    return _normalize_price_frame(history, is_intraday=interval != "1d")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _macd_histogram(series: pd.Series) -> pd.Series:
    macd_line = _ema(series, 12) - _ema(series, 26)
    signal = _ema(macd_line, 9)
    return macd_line - signal


def _obv(frame: pd.DataFrame) -> pd.Series:
    direction = np.sign(frame["Close"].diff().fillna(0))
    return (direction * frame["Volume"]).fillna(0).cumsum()


def _percentile_rank(series: pd.Series, value: float | None) -> float | None:
    clean = series.dropna()
    if value is None or clean.empty:
        return None
    return float((clean <= value).mean() * 100)


def _zscore_latest(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 20:
        return None
    std = clean.std(ddof=0)
    if not std:
        return 0.0
    return float((clean.iloc[-1] - clean.mean()) / std)


def _hurst_proxy(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 30:
        return None
    value = clean.autocorr(lag=1)
    return None if value is None or isnan(value) else float(0.5 + (value / 2))


def _resample_to_4h(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.set_index("Datetime").sort_index().resample("4H").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return data.dropna(subset=["Open", "Close"]).reset_index()


def _find_recent_swings(frame: pd.DataFrame, column: str, largest: bool, count: int = 10) -> list[float]:
    if len(frame) < 5:
        return []
    rolling = frame[column].rolling(window=5, center=True)
    markers = rolling.max() if largest else rolling.min()
    if largest:
        selected = frame.loc[frame[column] >= markers.fillna(np.inf), column]
    else:
        selected = frame.loc[frame[column] <= markers.fillna(-np.inf), column]
    return [float(value) for value in selected.dropna().tail(count).tolist()]


def _price_near_level(price: float, levels: list[float], tolerance_pct: float = 0.02) -> bool:
    return any(abs(price - level) / max(level, 1e-9) <= tolerance_pct for level in levels)


def _detect_gap_bias(frame: pd.DataFrame) -> dict[str, bool]:
    if len(frame) < 15:
        return {"gap_support_below": False, "gap_resistance_above": False}
    recent = frame.tail(30).copy()
    recent["prev_high"] = recent["High"].shift(1)
    recent["prev_low"] = recent["Low"].shift(1)
    return {
        "gap_support_below": bool(((recent["Low"] > recent["prev_high"]) & recent["prev_high"].notna()).any()),
        "gap_resistance_above": bool(((recent["High"] < recent["prev_low"]) & recent["prev_low"].notna()).any()),
    }


def _evaluate_volatility_regime(daily: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    atr_20 = _atr(daily, 20)
    hv_20 = daily["Close"].pct_change().rolling(20).std()
    current_atr = _safe_float(atr_20.iloc[-1])
    atr_200_avg = _safe_float(atr_20.tail(200).mean())
    ratio = (current_atr / atr_200_avg) if current_atr and atr_200_avg else None
    if ratio is None:
        regime = "unknown"
    elif ratio > 1.3:
        regime = "high"
    elif ratio < 0.7:
        regime = "low"
    else:
        regime = "normal"
    return regime, {
        "atr_20": current_atr,
        "historical_vol_20": _safe_float(hv_20.iloc[-1]),
        "atr_200_avg": atr_200_avg,
        "atr_ratio_vs_200": ratio,
    }


def _evaluate_bbw(daily: pd.DataFrame) -> tuple[float, str, dict[str, Any]]:
    sma = daily["Close"].rolling(20).mean()
    std = daily["Close"].rolling(20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    bbw = (upper - lower) / sma.replace(0, np.nan)
    current_bbw = _safe_float(bbw.iloc[-1])
    percentile = _percentile_rank(bbw, current_bbw)
    middle = _safe_float(sma.iloc[-1])
    price = _safe_float(daily["Close"].iloc[-1]) or 0.0
    middle_distance = abs(price - middle) / middle if middle else 1.0
    status = "normal"
    score = 0.35
    if percentile is not None and percentile < 20:
        status = "compressed"
        score = 0.55 + max(0.0, (20 - percentile) / 40)
        if middle_distance <= 0.02:
            score += 0.15
    elif percentile is not None and percentile > 80:
        status = "expanded"
        score = 0.2
    return _clip(score), status, {
        "bbw": current_bbw,
        "bbw_percentile": percentile,
        "bb_middle": middle,
        "bb_upper": _safe_float(upper.iloc[-1]),
        "bb_lower": _safe_float(lower.iloc[-1]),
    }


def _evaluate_momentum(daily: pd.DataFrame, spy_daily: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    close = daily["Close"]
    momentum_5d = _safe_float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) if len(close) >= 6 else None
    spy_close = spy_daily["Close"]
    spy_momentum = _safe_float((spy_close.iloc[-1] - spy_close.iloc[-6]) / spy_close.iloc[-6]) if len(spy_close) >= 6 else None
    rs_value = (momentum_5d - spy_momentum) if momentum_5d is not None and spy_momentum is not None else None
    rsi_value = _safe_float(_rsi(close).iloc[-1])
    macd_hist = _macd_histogram(close)
    hist_value = _safe_float(macd_hist.iloc[-1])
    hist_prev = _safe_float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else None

    score = 0.1
    if momentum_5d is not None:
        score += 0.5 if momentum_5d > 0.05 else 0.3 if momentum_5d > 0.02 else 0.0
    if rsi_value is not None:
        score += 0.4 if rsi_value > 70 else 0.7 if rsi_value > 60 else 0.45 if rsi_value > 50 else 0.0
    if hist_value is not None and hist_value > 0:
        score += 0.8 if hist_prev is not None and hist_value > hist_prev else 0.55
    if rs_value is not None and rs_value > 0:
        score += 0.6

    return _clip(score / 2.6), {
        "momentum_5d": momentum_5d,
        "rsi_14": rsi_value,
        "macd_histogram": hist_value,
        "relative_strength_vs_spy": rs_value,
    }


def _evaluate_volume(daily: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    volume = daily["Volume"]
    avg_20 = volume.rolling(20).mean()
    vol_ratio = _safe_float(volume.iloc[-1] / avg_20.iloc[-1]) if len(volume) >= 20 and avg_20.iloc[-1] else None
    obv = _obv(daily)
    obv_up = bool(len(obv) >= 5 and obv.iloc[-1] > obv.iloc[-5])
    if vol_ratio is None:
        score = 0.2
    elif vol_ratio >= 2.0:
        score = 1.0
    elif vol_ratio >= 1.5:
        score = 0.8
    elif vol_ratio >= 1.2:
        score = 0.6
    elif vol_ratio >= 1.0:
        score = 0.4
    else:
        score = 0.2
    if obv_up:
        score += 0.1
    recent = volume.tail(5).mean() if len(volume) >= 5 else None
    prior = volume.tail(10).head(5).mean() if len(volume) >= 10 else None
    return _clip(score), {
        "volume_ratio_20d": vol_ratio,
        "volume_trend_up_5d": bool(recent and prior and recent > prior),
        "obv_accumulating": obv_up,
        "volume_rate_of_change": _safe_float((volume.iloc[-1] - avg_20.iloc[-1]) / avg_20.iloc[-1]) if len(volume) >= 20 and avg_20.iloc[-1] else None,
    }


def _evaluate_confluence(daily: pd.DataFrame, four_hour: pd.DataFrame, hourly: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    frames = {"1d": daily, "4h": four_hour, "1h": hourly}
    price = _safe_float(daily["Close"].iloc[-1]) or 0.0
    score = 0.15
    metrics: dict[str, Any] = {"timeframes": {}}
    above_all = True

    for label, frame in frames.items():
        if frame.empty or len(frame) < 50:
            metrics["timeframes"][label] = {"above_200_sma": None, "sma_200": None}
            above_all = False
            continue
        sma_200 = _safe_float(frame["Close"].rolling(200).mean().iloc[-1])
        current_price = _safe_float(frame["Close"].iloc[-1])
        above = current_price is not None and sma_200 is not None and current_price > sma_200
        metrics["timeframes"][label] = {"above_200_sma": above, "sma_200": sma_200}
        above_all = above_all and bool(above)

    if above_all:
        score += 0.6
    elif metrics["timeframes"].get("1d", {}).get("above_200_sma") is False:
        score -= 0.2

    swing_lows = _find_recent_swings(daily, "Low", largest=False)
    swing_highs = _find_recent_swings(daily, "High", largest=True)
    support_bounce = _price_near_level(price, swing_lows) and len(daily) >= 2 and daily["Close"].iloc[-1] > daily["Close"].iloc[-2]
    resistance_break = bool(len(daily) >= 2 and price >= daily["High"].tail(10).max() and _price_near_level(price, swing_highs))
    if support_bounce:
        score += 0.3
    if resistance_break:
        score += 0.4

    atr_20 = _atr(daily, 20)
    atr_value = _safe_float(atr_20.iloc[-1]) or 0.0
    lower = price - atr_value
    upper = price + atr_value
    upper_third = price >= lower + ((upper - lower) * (2 / 3))
    metrics.update(
        {
            "support_bounce": support_bounce,
            "resistance_break": resistance_break,
            **_detect_gap_bias(daily),
            "atr_range_position": "upper" if upper_third else "middle_or_lower",
        }
    )
    return _clip(score), metrics


def _evaluate_mean_reversion(daily: pd.DataFrame, volatility_regime: str) -> dict[str, Any]:
    close = daily["Close"]
    price = _safe_float(close.iloc[-1]) or 0.0
    high_50 = _safe_float(close.tail(50).max())
    low_50 = _safe_float(close.tail(50).min())
    high_20 = _safe_float(close.tail(20).max())
    low_20 = _safe_float(close.tail(20).min())
    high_5 = _safe_float(close.tail(5).max())
    low_5 = _safe_float(close.tail(5).min())
    rsi_value = _safe_float(_rsi(close).iloc[-1])
    momentum_5d = _safe_float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]) if len(close) >= 6 else None
    mean_reversion_setup = bool(low_20 and price <= low_20 * 1.01 and volatility_regime == "low")
    pullback_setup = bool(high_50 and price >= high_50 * 0.99 and momentum_5d is not None and momentum_5d < 0.02)
    range_bound = bool(rsi_value is not None and 30 <= rsi_value <= 70 and high_20 and low_20 and low_20 < price < high_20)
    score = 0.8 if mean_reversion_setup else 0.65 if pullback_setup else 0.5 if range_bound else 0.2
    return {
        "mean_reversion_score": _clip(score),
        "mean_reversion_setup": mean_reversion_setup,
        "pullback_setup": pullback_setup,
        "range_bound": range_bound,
        "high_50": high_50,
        "low_50": low_50,
        "high_20": high_20,
        "low_20": low_20,
        "high_5": high_5,
        "low_5": low_5,
    }


def _evaluate_regime(daily: pd.DataFrame) -> tuple[str, dict[str, float | None]]:
    price = _safe_float(daily["Close"].iloc[-1])
    ema_50 = _safe_float(_ema(daily["Close"], 50).iloc[-1])
    ema_200 = _safe_float(_ema(daily["Close"], 200).iloc[-1])
    if price is None or ema_50 is None or ema_200 is None:
        regime = "choppy"
    elif price > ema_200 and price > ema_50:
        regime = "uptrend"
    elif price < ema_200 and price < ema_50:
        regime = "downtrend"
    else:
        regime = "choppy"
    return regime, {"ema_50": ema_50, "ema_200": ema_200}


def _evaluate_market_structure(daily: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    returns = daily["Close"].pct_change()
    rolling_vol = returns.rolling(20).std()
    hurst = _hurst_proxy(returns)
    structure = "trending" if hurst is not None and hurst > 0.5 else "mean_reverting"
    return structure, {
        "rolling_vol_20": _safe_float(rolling_vol.iloc[-1]),
        "volatility_increasing": bool(len(rolling_vol.dropna()) >= 5 and rolling_vol.dropna().iloc[-1] > rolling_vol.dropna().iloc[-5]),
        "hurst_proxy": hurst,
    }


def _evaluate_significance(daily: pd.DataFrame) -> tuple[float, float, dict[str, Any]]:
    momentum_series = daily["Close"].pct_change(5).tail(100)
    volume_ratio_series = (daily["Volume"] / daily["Volume"].rolling(20).mean()).tail(100)
    momentum_z = _zscore_latest(momentum_series)
    volume_z = _zscore_latest(volume_ratio_series)
    significant = bool(momentum_z is not None and momentum_z >= 2.0 and volume_z is not None and volume_z >= 2.0)
    return (1.0 if significant else 0.35), (0.10 if significant else -0.05), {
        "momentum_zscore": momentum_z,
        "volume_zscore": volume_z,
        "significant": significant,
    }


def _extract_earnings_date(calendar: Any) -> datetime | None:
    try:
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            candidate = pd.to_datetime(calendar.iloc[:, 0], errors="coerce").dropna()
            if not candidate.empty:
                return candidate.iloc[0].to_pydatetime()
        if isinstance(calendar, dict):
            for value in calendar.values():
                if isinstance(value, (list, tuple)) and value:
                    candidate = pd.to_datetime(value[0], errors="coerce")
                else:
                    candidate = pd.to_datetime(value, errors="coerce")
                if pd.notna(candidate):
                    return candidate.to_pydatetime()
    except Exception:
        return None
    return None


def _evaluate_catalyst(ticker: str) -> CatalystResult:
    try:
        stock = yf.Ticker(ticker)
    except Exception:
        return CatalystResult(0.3, "none", "No catalyst data available.", "none", "fallback")
    try:
        earnings_date = _extract_earnings_date(stock.calendar)
    except Exception:
        earnings_date = None
    if earnings_date is not None:
        days = (earnings_date.date() - datetime.now(PKT).date()).days
        if days <= 1:
            return CatalystResult(0.9, "earnings", "Earnings are scheduled today or tomorrow.", "today_or_tomorrow", "yahoo_calendar")
        if days <= 7:
            return CatalystResult(0.8, "earnings", "Earnings are scheduled this week.", "this_week", "yahoo_calendar")
        if days <= 14:
            return CatalystResult(0.6, "earnings", "Earnings are scheduled next week.", "next_week", "yahoo_calendar")
    try:
        news_items = getattr(stock, "news", []) or []
    except Exception:
        news_items = []
    if news_items:
        title = news_items[0].get("title", "Recent news catalyst detected.")
        return CatalystResult(0.55, "news", title, "recent", "yahoo_news")
    return CatalystResult(0.3, "none", "No clear catalyst detected from free sources.", "none", "fallback")


def _screen_shariah(stock_data: dict) -> ShariahScreenResult:
    violations: list[str] = []
    warnings: list[str] = []
    sector = stock_data.get("sector")
    industry = stock_data.get("industry")
    searchable = f"{sector or ''} {industry or ''}".lower()
    if any(keyword in searchable for keyword in SECTOR_EXCLUSION_KEYWORDS):
        violations.append("sector_exclusion")

    market_cap = _safe_float(stock_data.get("market_cap"))
    total_debt = _safe_float(stock_data.get("total_debt"))
    debt_ratio = None
    if market_cap and total_debt is not None:
        debt_ratio = total_debt / market_cap
        if debt_ratio >= 0.33:
            violations.append("debt_ratio_exceeds_33_percent")
    else:
        warnings.append("Debt or market-cap data was unavailable, so the ratio check is best-effort.")

    sec_income_data = stock_data.get("sec_income_data", {})
    income_note = sec_income_data.get("message", "Income screening data unavailable.")
    income_fact = sec_income_data.get("selected_non_core_income_fact")
    revenue_fact = sec_income_data.get("revenue_fact")
    if income_fact and revenue_fact and revenue_fact.get("value"):
        ratio = income_fact["value"] / revenue_fact["value"]
        if ratio > 0.05:
            warnings.append("Possible non-compliant income appears material in the latest SEC filing.")
            income_note = f"Possible non-core income ratio is {ratio:.2%}."
    else:
        warnings.append("Income screening is incomplete and should be treated as a flag, not a hard fail.")

    passed = len(violations) == 0
    return ShariahScreenResult(
        passed=passed,
        status="PASS" if passed else "FAIL",
        violations=violations,
        warnings=warnings,
        debt_to_market_cap=debt_ratio,
        sector=sector,
        industry=industry,
        income_screen_note=income_note,
    )


def _build_weights(regime: str) -> tuple[dict[str, float], float]:
    weights = {"momentum": 0.25, "volume": 0.15, "confluence": 0.20, "bbw": 0.15, "catalyst": 0.15}
    regime_bonus = -0.05
    if regime == "uptrend":
        weights["momentum"] = 0.30
        weights["confluence"] = 0.15
        regime_bonus = 0.05
    elif regime == "downtrend":
        weights["momentum"] = 0.18
        weights["confluence"] = 0.27
        regime_bonus = -0.02
    return weights, regime_bonus


def _classify_tier(composite: float, shariah_passed: bool) -> str:
    if not shariah_passed or composite <= 0:
        return "Disqualified"
    if composite >= 0.75:
        return "Tier 1"
    if composite >= 0.55:
        return "Tier 2"
    if composite >= 0.35:
        return "Tier 3"
    return "Disqualified"


def _classify_signal(
    composite: float,
    shariah: ShariahScreenResult,
    regime: str,
    confluence_score: float,
    significance_metrics: dict[str, Any],
) -> str:
    if not shariah.passed:
        return "DISQUALIFIED"
    if significance_metrics.get("significant") and composite >= 0.75 and regime == "uptrend" and confluence_score >= 0.55:
        return "BUY"
    if composite >= 0.55:
        return "WATCH"
    if composite >= 0.35:
        return "SKIP"
    return "DISQUALIFIED"


def _build_narratives(
    scores: ScoreBreakdown,
    catalyst: CatalystResult,
    regime: str,
    shariah: ShariahScreenResult,
    technical_metrics: dict[str, Any],
) -> dict[str, str]:
    momentum = technical_metrics["momentum"]
    volume = technical_metrics["volume"]
    bbw = technical_metrics["bbw"]
    significance = technical_metrics["significance"]
    return {
        "momentum": f"Momentum score {scores.momentum:.2f}. 5d momentum is {momentum['momentum_5d']:.2%} with RSI {momentum['rsi_14']:.1f}." if momentum["momentum_5d"] is not None and momentum["rsi_14"] is not None else f"Momentum score {scores.momentum:.2f}.",
        "volume": f"Volume score {scores.volume:.2f}. Current volume is {volume['volume_ratio_20d']:.2f}x the 20-day average." if volume["volume_ratio_20d"] is not None else f"Volume score {scores.volume:.2f}.",
        "confluence": f"Confluence score {scores.confluence:.2f}. Multi-timeframe structure and support/resistance were combined.",
        "bbw": f"BBW score {scores.bbw:.2f}. Current BB width sits near the {bbw['bbw_percentile']:.1f} percentile." if bbw["bbw_percentile"] is not None else f"BBW score {scores.bbw:.2f}.",
        "catalyst": f"Catalyst score {scores.catalyst:.2f}. {catalyst.summary}",
        "regime": f"Regime is {regime}. Market structure is {technical_metrics['market_structure']['label']}.",
        "significance": "Signal quality is high because momentum and volume are both 2+ standard deviations above mean." if significance.get("significant") else "Signal quality is below the strict 2-sigma threshold.",
        "shariah": f"Shariah status is {shariah.status}. " + ("; ".join(shariah.warnings) if shariah.warnings else "No hard violations detected."),
    }


def analyze_ticker(
    ticker: str,
    *,
    stock_data_fetcher: Callable[[str], dict] = get_stock_data,
    daily_history_fetcher: Callable[[str], pd.DataFrame | None] | None = None,
    intraday_history_fetcher: Callable[[str], pd.DataFrame | None] | None = None,
    spy_history_fetcher: Callable[[], pd.DataFrame | None] | None = None,
    catalyst_fetcher: Callable[[str], CatalystResult] | None = None,
) -> AutonomousAnalysisResult:
    symbol = (ticker or "").strip().upper()
    analyzed_at = _now_pkt_iso()
    if not symbol:
        shariah = ShariahScreenResult(True, "PASS", [], ["Ticker was empty."], None, None, None, "No income data.")
        catalyst = CatalystResult(0.3, "none", "No catalyst data available.", "none", "fallback")
        scores = ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.3, 0.0, -0.05, -0.05, 0.0, {})
        return AutonomousAnalysisResult(symbol, analyzed_at, "error", "Ticker not found.", None, None, None, "unknown", "unknown", "choppy", "mean_reverting", "Disqualified", "DISQUALIFIED", shariah, catalyst, scores, {}, {"summary": "Ticker was empty."}, ["Ticker not found."])

    daily_history_fetcher = daily_history_fetcher or (lambda value: _download_history(value, "1y", "1d"))
    intraday_history_fetcher = intraday_history_fetcher or (lambda value: _download_history(value, "6mo", "1h"))
    spy_history_fetcher = spy_history_fetcher or (lambda: _download_history("SPY", "1y", "1d"))
    catalyst_fetcher = catalyst_fetcher or _evaluate_catalyst

    daily = daily_history_fetcher(symbol)
    intraday = intraday_history_fetcher(symbol)
    spy_daily = spy_history_fetcher()
    stock_data = stock_data_fetcher(symbol)
    shariah = _screen_shariah(stock_data)
    catalyst = catalyst_fetcher(symbol)

    if daily is None or daily.empty or spy_daily is None or spy_daily.empty:
        scores = ScoreBreakdown(0.0, 0.0, 0.0, 0.0, round(catalyst.score, 2), 0.0, -0.05, -0.05, 0.0, {})
        return AutonomousAnalysisResult(symbol, analyzed_at, "error", "Insufficient data.", None, None, None, "unknown", "unknown", "choppy", "mean_reverting", "Disqualified", "DISQUALIFIED", shariah, catalyst, scores, {}, {"summary": "Price history was unavailable or incomplete."}, ["Insufficient data"])

    intraday = intraday if intraday is not None else pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    four_hour = _resample_to_4h(intraday) if not intraday.empty else pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

    volatility_regime, volatility_metrics = _evaluate_volatility_regime(daily)
    bbw_score, compression_status, bbw_metrics = _evaluate_bbw(daily)
    momentum_score, momentum_metrics = _evaluate_momentum(daily, spy_daily)
    volume_score, volume_metrics = _evaluate_volume(daily)
    confluence_score, confluence_metrics = _evaluate_confluence(daily, four_hour, intraday)
    mean_reversion_metrics = _evaluate_mean_reversion(daily, volatility_regime)
    regime, regime_metrics = _evaluate_regime(daily)
    market_structure, market_structure_metrics = _evaluate_market_structure(daily)
    significance_score, significance_bonus, significance_metrics = _evaluate_significance(daily)

    weights, regime_bonus = _build_weights(regime)
    composite = (
        momentum_score * weights["momentum"]
        + volume_score * weights["volume"]
        + confluence_score * weights["confluence"]
        + bbw_score * weights["bbw"]
        + catalyst.score * weights["catalyst"]
        + regime_bonus
        + significance_bonus
    )
    composite = 0.0 if not shariah.passed else _clip(composite)
    scores = ScoreBreakdown(
        momentum=round(momentum_score, 2),
        volume=round(volume_score, 2),
        confluence=round(confluence_score, 2),
        bbw=round(bbw_score, 2),
        catalyst=round(catalyst.score, 2),
        significance=round(significance_score, 2),
        regime_bonus=round(regime_bonus, 2),
        significance_bonus=round(significance_bonus, 2),
        composite=round(composite, 2),
        weights=weights,
    )
    technical_metrics = {
        "volatility": volatility_metrics,
        "bbw": bbw_metrics,
        "momentum": momentum_metrics,
        "volume": volume_metrics,
        "confluence": confluence_metrics,
        "mean_reversion": mean_reversion_metrics,
        "regime": regime_metrics,
        "market_structure": {"label": market_structure, **market_structure_metrics},
        "significance": significance_metrics,
    }
    tier = _classify_tier(scores.composite, shariah.passed)
    trade_signal = _classify_signal(scores.composite, shariah, regime, scores.confluence, significance_metrics)
    flags = [warning for warning in shariah.warnings]
    if intraday.empty:
        flags.append("Partial intraday data.")
    return AutonomousAnalysisResult(
        ticker=symbol,
        analyzed_at_pk_time=analyzed_at,
        status="ok",
        error=None,
        latest_price=_safe_float(daily["Close"].iloc[-1]),
        price_change_5d_pct=momentum_metrics["momentum_5d"],
        relative_strength_vs_spy=momentum_metrics["relative_strength_vs_spy"],
        volatility_regime=volatility_regime,
        compression_status=compression_status,
        regime=regime,
        market_structure=market_structure,
        tier=tier,
        trade_signal=trade_signal,
        shariah=shariah,
        catalyst=catalyst,
        scores=scores,
        technical_metrics=technical_metrics,
        narratives=_build_narratives(scores, catalyst, regime, shariah, technical_metrics),
        flags=flags,
    )


def analyze_tickers(
    tickers: list[str],
    *,
    max_workers: int = 8,
    analyzer: Callable[[str], AutonomousAnalysisResult] = analyze_ticker,
) -> list[AutonomousAnalysisResult]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        symbol = (ticker or "").strip().upper()
        if symbol and symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    if not ordered:
        return []

    results: list[AutonomousAnalysisResult] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(ordered)))) as executor:
        futures = {executor.submit(analyzer, ticker): ticker for ticker in ordered}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                results.append(analyze_ticker(futures[future], daily_history_fetcher=lambda _: None, intraday_history_fetcher=lambda _: None, spy_history_fetcher=lambda: None))

    tier_order = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2, "Disqualified": 3}
    results.sort(key=lambda item: (tier_order.get(item.tier, 4), -item.scores.composite, item.ticker))
    return results


def export_autonomous_results_json(results: list[AutonomousAnalysisResult | dict[str, Any]]) -> str:
    payload = [result.to_dict() if hasattr(result, "to_dict") else result for result in results]
    return json.dumps(payload, indent=2)


def parse_ticker_text(text: str) -> list[str]:
    cleaned = (text or "").replace(",", "\n").replace(";", "\n")
    return [item.strip().upper() for item in cleaned.splitlines() if item.strip()]


def parse_ticker_csv(content: bytes) -> list[str]:
    if not content:
        return []
    frame = pd.read_csv(StringIO(content.decode("utf-8")))
    for column in frame.columns:
        if str(column).lower() in {"ticker", "symbol"}:
            return [str(value).strip().upper() for value in frame[column].dropna().tolist() if str(value).strip()]
    first_column = frame.columns[0] if len(frame.columns) else None
    if first_column is None:
        return []
    return [str(value).strip().upper() for value in frame[first_column].dropna().tolist() if str(value).strip()]
