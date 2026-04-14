"""SEC filing helpers for a best-effort income screen.

This module uses free SEC EDGAR JSON APIs. It is intentionally simple and
honest: it only reports what it can find from the filing data and does not
pretend to know unavailable details.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import requests


SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
NON_CORE_INCOME_CONCEPT_GROUPS = {
    "interest_income": [
        "InvestmentIncomeInterest",
        "InterestAndOtherIncome",
        "InterestIncomeOther",
        "InterestIncomeOperating",
        "InterestAndDividendIncomeOperating",
        "InterestIncomeExpenseNonoperatingNet",
    ],
    "dividend_income": [
        "DividendIncome",
        "InvestmentIncomeDividend",
        "InterestAndDividendIncomeOperating",
        "InvestmentIncomeInterestAndDividend",
    ],
    "investment_income": [
        "InvestmentIncomeInterestAndDividend",
        "InvestmentIncome",
        "OtherThanTemporaryImpairmentLossesInvestmentsPortionRecognizedInEarningsNet",
    ],
    "other_non_operating_income": [
        "OtherNonoperatingIncome",
        "NonoperatingIncomeExpense",
        "OtherNonoperatingIncomeExpense",
    ],
}


def _get_headers() -> dict:
    """Build SEC request headers.

    The SEC asks automated tools to send a declared user agent. Users can set
    SEC_USER_AGENT later without changing the code.
    """
    user_agent = os.getenv(
        "SEC_USER_AGENT",
        "ShariahStockScreenerMVP/1.0 contact@example.com",
    )
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _get_json(url: str) -> dict | list:
    """Fetch JSON safely from the SEC."""
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, headers=_get_headers(), timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)

    raise last_error


def _pick_latest_fact(unit_items: list[dict[str, Any]]) -> dict | None:
    """Pick the most recent usable fact from a list of SEC fact entries."""
    dated_items = [
        item
        for item in unit_items
        if isinstance(item, dict) and item.get("val") is not None and item.get("end")
    ]
    if not dated_items:
        return None

    sorted_items = sorted(
        dated_items,
        key=lambda item: (item.get("fy", 0), item.get("end", ""), item.get("filed", "")),
        reverse=True,
    )
    return sorted_items[0]


def _get_fact_value(company_facts: dict, concept_names: list[str]) -> dict | None:
    """Return the first matching fact found from the requested concept list."""
    facts = company_facts.get("facts", {})
    us_gaap = facts.get("us-gaap", {})

    for concept_name in concept_names:
        concept = us_gaap.get(concept_name)
        if not concept:
            continue

        units = concept.get("units", {})
        usd_items = units.get("USD", [])
        latest_fact = _pick_latest_fact(usd_items)
        if latest_fact:
            return {
                "concept": concept_name,
                "label": concept.get("label") or concept_name,
                "value": latest_fact.get("val"),
                "fy": latest_fact.get("fy"),
                "end": latest_fact.get("end"),
                "filed": latest_fact.get("filed"),
                "form": latest_fact.get("form"),
            }

    return None


def _get_matching_facts(
    company_facts: dict,
    concept_names: list[str],
    category: str,
) -> list[dict]:
    """Return all matching latest facts for a concept list, without duplicates."""
    matches = []
    seen_concepts = set()

    for concept_name in concept_names:
        if concept_name in seen_concepts:
            continue

        fact = _get_fact_value(company_facts, [concept_name])
        if not fact:
            continue

        fact["category"] = category
        matches.append(fact)
        seen_concepts.add(concept_name)

    return matches


def _get_cik_for_ticker(ticker: str) -> dict:
    """Look up the SEC CIK for a ticker symbol."""
    ticker_map = _get_json(SEC_TICKER_URL)

    if not isinstance(ticker_map, dict):
        return {"status": "error", "message": "Unexpected SEC ticker lookup response."}

    search_ticker = ticker.upper()
    for item in ticker_map.values():
        if str(item.get("ticker", "")).upper() == search_ticker:
            cik_str = str(item.get("cik_str", "")).strip()
            if not cik_str:
                continue
            return {
                "status": "ok",
                "cik": cik_str.zfill(10),
                "company_name": item.get("title"),
            }

    return {
        "status": "error",
        "message": f"SEC could not find a CIK for ticker '{ticker}'.",
    }


@lru_cache(maxsize=128)
def get_sec_income_data(ticker: str) -> dict:
    """Fetch best-effort income-screen data from the SEC."""
    limitations = [
        "SEC income screening is best-effort and depends on which XBRL facts the company filed.",
        "Some companies do not expose a clean interest-income or non-core-income fact in a way that can be screened automatically.",
        "A missing SEC fact does not prove the company has zero non-compliant income.",
        "Some SEC concepts can overlap, so this MVP prioritizes one best candidate fact instead of summing everything.",
    ]

    try:
        cik_result = _get_cik_for_ticker(ticker)
        if cik_result["status"] != "ok":
            return {
                "status": "error",
                "message": cik_result["message"],
                "limitations": limitations,
            }

        company_facts = _get_json(SEC_COMPANY_FACTS_URL.format(cik=cik_result["cik"]))
    except Exception as error:
        return {
            "status": "error",
            "message": (
                "Could not fetch SEC filing data right now. "
                f"Technical detail: {error}"
            ),
            "limitations": limitations,
        }

    revenue_fact = _get_fact_value(
        company_facts,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
    )

    non_core_income_facts = []
    for category, concept_names in NON_CORE_INCOME_CONCEPT_GROUPS.items():
        non_core_income_facts.extend(
            _get_matching_facts(company_facts, concept_names, category)
        )

    selected_non_core_income_fact = non_core_income_facts[0] if non_core_income_facts else None

    return {
        "status": "ok",
        "message": "SEC filing data fetched successfully.",
        "limitations": limitations,
        "cik": cik_result["cik"],
        "sec_company_name": cik_result.get("company_name"),
        "revenue_fact": revenue_fact,
        "selected_non_core_income_fact": selected_non_core_income_fact,
        "non_core_income_facts": non_core_income_facts,
    }
