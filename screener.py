"""Screening logic for the Shariah stock screener MVP."""

from __future__ import annotations

from utils import combine_notes, create_plain_english_explanation


def _check_business_activity(stock_data: dict, methodology: dict) -> dict:
    """Run a simple keyword-based business activity screen."""
    sector = (stock_data.get("sector") or "").lower()
    industry = (stock_data.get("industry") or "").lower()
    searchable_text = f"{sector} {industry}".strip()

    if not searchable_text:
        return {
            "status": "unavailable",
            "note": (
                "Sector and industry data were not available, so the placeholder business "
                "activity screen could not be completed."
            ),
        }

    for keyword in methodology["business_screen"]["prohibited_keywords"]:
        if keyword in searchable_text:
            return {
                "status": "fail",
                "note": (
                    f"The company was flagged because its sector or industry appears to "
                    f"match the prohibited keyword '{keyword}'. This is a simple placeholder check."
                ),
            }

    return {
        "status": "pass",
        "note": (
            "The sector and industry did not clearly match the prohibited keyword list. "
            "This is only a limited placeholder business activity screen, not a detailed "
            "revenue-based analysis."
        ),
    }


def _calculate_ratio(stock_data: dict, ratio_rule: dict) -> dict:
    """Calculate one financial ratio using simple threshold logic."""
    numerator = stock_data.get(ratio_rule["numerator"])
    denominator = stock_data.get(ratio_rule["denominator"])

    if numerator is None or denominator is None:
        return {
            "key": ratio_rule["key"],
            "label": ratio_rule["label"],
            "value": None,
            "status": "unavailable",
            "threshold_label": _build_threshold_label(ratio_rule),
            "note": "Required data for this ratio was missing.",
        }

    if denominator == 0:
        return {
            "key": ratio_rule["key"],
            "label": ratio_rule["label"],
            "value": None,
            "status": "unavailable",
            "threshold_label": _build_threshold_label(ratio_rule),
            "note": "The denominator was zero, so this ratio could not be calculated safely.",
        }

    value = numerator / denominator

    if "max_threshold" in ratio_rule:
        status = "pass" if value <= ratio_rule["max_threshold"] else "fail"
    elif "min_threshold" in ratio_rule:
        status = "pass" if value >= ratio_rule["min_threshold"] else "fail"
    else:
        status = "unavailable"

    return {
        "key": ratio_rule["key"],
        "label": ratio_rule["label"],
        "value": value,
        "status": status,
        "threshold_label": _build_threshold_label(ratio_rule),
        "note": "Ratio calculated successfully.",
    }


def _build_threshold_label(ratio_rule: dict) -> str:
    """Convert threshold rules into a readable label."""
    if "max_threshold" in ratio_rule:
        return f"Must be <= {ratio_rule['max_threshold']:.0%}"
    if "min_threshold" in ratio_rule:
        return f"Must be >= {ratio_rule['min_threshold']:.0%}"
    return "No threshold set"


def _check_financial_screen(stock_data: dict, methodology: dict) -> dict:
    """Run all configured financial ratio checks."""
    ratio_results = [
        _calculate_ratio(stock_data, ratio_rule)
        for ratio_rule in methodology["financial_screen"]["ratios"]
    ]

    has_fail = any(item["status"] == "fail" for item in ratio_results)
    has_unavailable = any(item["status"] == "unavailable" for item in ratio_results)

    if has_fail:
        status = "fail"
        note = "At least one financial ratio failed the methodology thresholds."
    elif has_unavailable:
        status = "unavailable"
        note = "Some financial ratio checks could not be completed because data was missing."
    else:
        status = "pass"
        note = "All available financial ratio checks passed the methodology thresholds."

    return {
        "status": status,
        "note": note,
        "ratio_results": ratio_results,
    }


def _check_income_screen(stock_data: dict, methodology: dict) -> dict:
    """Run a best-effort income screen using SEC XBRL facts."""
    sec_income_data = stock_data.get("sec_income_data", {})
    if sec_income_data.get("status") != "ok":
        return {
            "status": "unavailable",
            "note": sec_income_data.get(
                "message",
                "SEC filing data was not available for the income screen.",
            ),
            "selected_non_core_income_fact": None,
            "non_core_income_facts": [],
            "revenue_fact": None,
            "non_core_income_ratio": None,
            "threshold_label": (
                f"Must be <= {methodology['income_screen']['max_non_core_income_ratio']:.0%}"
            ),
        }

    selected_non_core_income_fact = sec_income_data.get("selected_non_core_income_fact")
    non_core_income_facts = sec_income_data.get("non_core_income_facts", [])
    revenue_fact = sec_income_data.get("revenue_fact")
    threshold = methodology["income_screen"]["max_non_core_income_ratio"]

    if not selected_non_core_income_fact:
        return {
            "status": "unavailable",
            "note": (
                "No clear non-core-income fact was found in the recent SEC XBRL data. "
                "This does not prove the value is zero. It only means the parser could not "
                "find a clean reported line item."
            ),
            "selected_non_core_income_fact": None,
            "non_core_income_facts": [],
            "revenue_fact": revenue_fact,
            "non_core_income_ratio": None,
            "threshold_label": f"Must be <= {threshold:.0%}",
        }

    if not revenue_fact or not revenue_fact.get("value"):
        return {
            "status": "unavailable",
            "note": (
                "A possible non-core-income fact was found in the SEC data, but a usable "
                "revenue figure was not found, so the ratio could not be calculated safely."
            ),
            "selected_non_core_income_fact": selected_non_core_income_fact,
            "non_core_income_facts": non_core_income_facts,
            "revenue_fact": revenue_fact,
            "non_core_income_ratio": None,
            "threshold_label": f"Must be <= {threshold:.0%}",
        }

    non_core_income_ratio = selected_non_core_income_fact["value"] / revenue_fact["value"]
    status = "pass" if non_core_income_ratio <= threshold else "fail"

    if status == "pass":
        note = (
            "A usable SEC possible non-core-income fact and revenue fact were found, and the "
            "ratio was within the current methodology threshold."
        )
    else:
        note = (
            "A usable SEC possible non-core-income fact and revenue fact were found, and the "
            "ratio was above the current methodology threshold."
        )

    return {
        "status": status,
        "note": note,
        "selected_non_core_income_fact": selected_non_core_income_fact,
        "non_core_income_facts": non_core_income_facts,
        "revenue_fact": revenue_fact,
        "non_core_income_ratio": non_core_income_ratio,
        "threshold_label": f"Must be <= {threshold:.0%}",
    }


def screen_stock(stock_data: dict, methodology: dict) -> dict:
    """Run the full screening flow and return one simple result dictionary."""
    business_result = _check_business_activity(stock_data, methodology)
    financial_result = _check_financial_screen(stock_data, methodology)
    income_result = _check_income_screen(stock_data, methodology)

    if (
        business_result["status"] == "fail"
        or financial_result["status"] == "fail"
        or income_result["status"] == "fail"
    ):
        final_verdict = "Non-compliant"
    elif (
        business_result["status"] == "unavailable"
        or financial_result["status"] == "unavailable"
        or income_result["status"] == "unavailable"
    ):
        final_verdict = "Insufficient data"
    else:
        final_verdict = "Compliant"

    limitations = combine_notes(
        methodology["limitations"],
        stock_data.get("limitations", []),
    )

    return {
        "company": {
            "company_name": stock_data.get("company_name", "Unknown company"),
            "ticker": stock_data.get("ticker", ""),
            "sector": stock_data.get("sector"),
            "industry": stock_data.get("industry"),
            "market_cap": stock_data.get("market_cap"),
            "total_debt": stock_data.get("total_debt"),
            "cash": stock_data.get("cash"),
            "total_assets": stock_data.get("total_assets"),
            "current_assets": stock_data.get("current_assets"),
        },
        "methodology": {
            "name": methodology["name"],
            "description": methodology["description"],
        },
        "business_screen": business_result,
        "financial_screen": financial_result,
        "income_screen": income_result,
        "final_verdict": final_verdict,
        "plain_english_explanation": create_plain_english_explanation(
            stock_data=stock_data,
            business_result=business_result,
            financial_result=financial_result,
            final_verdict=final_verdict,
        ),
        "limitations": limitations,
    }
