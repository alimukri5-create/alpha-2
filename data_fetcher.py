"""Data fetching layer for stock information.

This module uses yfinance because it is beginner-friendly for a prototype.
It is not an official institutional data feed.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import yfinance as yf
from sec_parser import get_sec_income_data


def _pick_first_number(*values: Any) -> float | None:
    """Return the first usable numeric value from a list of possible fields."""
    for value in values:
        if isinstance(value, (int, float)) and value is not None:
            return float(value)
    return None


def _read_balance_sheet_value(stock: yf.Ticker, possible_labels: list[str]) -> float | None:
    """Try to read a value from the balance sheet using a few common row labels."""
    try:
        balance_sheet = stock.balance_sheet
    except Exception:
        return None

    if balance_sheet is None or getattr(balance_sheet, "empty", True):
        return None

    for label in possible_labels:
        if label in balance_sheet.index:
            series = balance_sheet.loc[label]
            for value in series.tolist():
                picked_value = _pick_first_number(value)
                if picked_value is not None:
                    return picked_value

    return None


def _safe_get_info(stock: yf.Ticker, max_attempts: int = 3) -> tuple[dict | None, str | None]:
    """Fetch Yahoo info with small retries for temporary rate limits."""
    last_error = None

    for attempt in range(max_attempts):
        try:
            info = stock.info
            if isinstance(info, dict) and info:
                return info, None
        except Exception as error:
            last_error = str(error)
            if "Too Many Requests" not in last_error and "Rate limited" not in last_error:
                break

        if attempt < max_attempts - 1:
            time.sleep(attempt + 1)

    return None, last_error


@lru_cache(maxsize=128)
def get_stock_data(ticker: str) -> dict:
    """Fetch raw company and financial fields for one ticker.

    The return shape is intentionally simple so the data provider can be replaced later.
    """
    limitations = [
        "Prototype source: Yahoo Finance via yfinance.",
        "Some tickers may have missing fields or inconsistent metadata.",
    ]

    try:
        stock = yf.Ticker(ticker)
        info, yahoo_error = _safe_get_info(stock)
    except Exception as error:
        return {
            "status": "error",
            "message": (
                f"Could not fetch data for ticker '{ticker}'. "
                f"Please check the ticker and try again. Technical detail: {error}"
            ),
            "limitations": limitations,
        }

    sec_income_data = get_sec_income_data(ticker)
    limitations.extend(sec_income_data.get("limitations", []))

    if not isinstance(info, dict) or not info:
        if yahoo_error and ("Too Many Requests" in yahoo_error or "Rate limited" in yahoo_error):
            limitations.append(
                "Yahoo Finance rate-limited this request. The app is showing whatever it could "
                "still collect from SEC data."
            )
            return {
                "status": "ok",
                "message": "Partial data fetched successfully.",
                "limitations": limitations,
                "data_source": "SEC EDGAR with Yahoo Finance fallback unavailable",
                "ticker": ticker,
                "company_name": sec_income_data.get("sec_company_name") or ticker,
                "sector": None,
                "industry": None,
                "market_cap": None,
                "total_debt": None,
                "cash": None,
                "total_assets": None,
                "current_assets": None,
                "sec_income_data": sec_income_data,
            }

        return {
            "status": "error",
            "message": (
                f"No data was returned for ticker '{ticker}'. "
                "Please try a different US-listed stock ticker."
            ),
            "limitations": limitations,
        }

    company_name = info.get("longName") or info.get("shortName") or ticker
    quote_type = info.get("quoteType")
    market_cap = _pick_first_number(info.get("marketCap"), info.get("enterpriseValue"))
    total_debt = _pick_first_number(info.get("totalDebt"))
    cash = _pick_first_number(
        info.get("totalCash"),
        info.get("cash"),
        info.get("cashAndCashEquivalents"),
    )
    total_assets = _pick_first_number(info.get("totalAssets"))
    current_assets = _pick_first_number(info.get("currentAssets"))

    if quote_type and str(quote_type).lower() not in {"equity", "stock"}:
        return {
            "status": "error",
            "message": (
                f"Ticker '{ticker}' does not appear to be a normal stock equity. "
                "Please try a US-listed company stock ticker."
            ),
            "limitations": limitations,
        }

    if market_cap is None:
        limitations.append("Market cap was missing, which may block some ratio checks.")

    if total_debt is None:
        total_debt = _read_balance_sheet_value(stock, ["Total Debt", "Current Debt And Capital Lease Obligation"])
    if cash is None:
        cash = _read_balance_sheet_value(stock, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
    if total_assets is None:
        total_assets = _read_balance_sheet_value(stock, ["Total Assets"])
    if current_assets is None:
        current_assets = _read_balance_sheet_value(stock, ["Current Assets"])

    if yahoo_error and ("Too Many Requests" in yahoo_error or "Rate limited" in yahoo_error):
        limitations.append(
            "Yahoo Finance briefly rate-limited this request, but the app recovered enough data to continue."
        )

    return {
        "status": "ok",
        "message": "Data fetched successfully.",
        "limitations": limitations,
        "data_source": "Yahoo Finance via yfinance",
        "ticker": ticker,
        "company_name": company_name,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "total_assets": total_assets,
        "current_assets": current_assets,
        "sec_income_data": sec_income_data,
    }
