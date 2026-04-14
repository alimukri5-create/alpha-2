"""Small helper functions for formatting and friendly messages."""

from __future__ import annotations


def clean_ticker(text: str) -> str:
    """Clean user input into a simple uppercase ticker."""
    return (text or "").strip().upper()


def format_number(value: float | None) -> str:
    """Format large numbers for display."""
    if value is None:
        return "Not available"

    absolute_value = abs(value)
    if absolute_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"${value / 1_000:.2f}K"
    return f"${value:,.2f}"


def format_percentage(value: float | None) -> str:
    """Format a ratio as a percentage."""
    if value is None:
        return "Not available"
    return f"{value:.2%}"


def get_status_label(status: str) -> str:
    """Convert internal status values into readable labels."""
    mapping = {
        "pass": "Pass",
        "fail": "Fail",
        "unavailable": "Unavailable",
    }
    return mapping.get(status, status.title())


def combine_notes(first_list: list[str], second_list: list[str]) -> list[str]:
    """Combine note lists without duplicates."""
    merged: list[str] = []
    for note in [*first_list, *second_list]:
        if note and note not in merged:
            merged.append(note)
    return merged


def create_plain_english_explanation(
    stock_data: dict,
    business_result: dict,
    financial_result: dict,
    final_verdict: str,
) -> str:
    """Create a simple beginner-friendly explanation."""
    company_name = stock_data.get("company_name", "This company")

    if final_verdict == "Non-compliant":
        return (
            f"{company_name} was marked as Non-compliant because it failed at least one "
            "part of the methodology. Please review the business activity, income screen, "
            "and financial ratio checks shown above."
        )

    if final_verdict == "Insufficient data":
        return (
            f"{company_name} could not be screened with confidence because some required "
            "data was missing. The app is showing what it could check, but it is not "
            "safe to give a full verdict from incomplete data."
        )

    if business_result["status"] == "pass" and financial_result["status"] == "pass":
        return (
            f"{company_name} passed this simple MVP methodology. That means its sector "
            "and industry did not trigger the placeholder prohibited keyword screen, and "
            "the available income and financial checks passed the configured thresholds."
        )

    return (
        f"{company_name} was reviewed using the current methodology, but you should read "
        "the detailed notes because this prototype uses limited data and simplified rules."
    )
