"""Methodology configuration for the Shariah stock screener MVP."""


def get_default_methodology() -> dict:
    """Return one simple, editable methodology for the MVP.

    All thresholds live here so they can be changed later without editing the UI.
    """
    return {
        "name": "Simple MVP Methodology (Prototype)",
        "description": (
            "This MVP uses a limited business activity screen based on sector and "
            "industry keywords, plus simple financial ratio checks using Yahoo Finance "
            "prototype data. It is a methodology-based tool, not a religious ruling."
        ),
        "business_screen": {
            "prohibited_keywords": [
                "bank",
                "banks",
                "insurance",
                "gambling",
                "casino",
                "alcohol",
                "beer",
                "brew",
                "tobacco",
                "cigarette",
                "adult",
                "porn",
                "weapons",
                "defense",
                "defence",
                "armaments",
                "mortgage",
                "lending",
                "interest",
                "credit services",
            ]
        },
        "financial_screen": {
            "ratios": [
                {
                    "key": "debt_to_market_cap",
                    "label": "Total debt / Market cap",
                    "numerator": "total_debt",
                    "denominator": "market_cap",
                    "max_threshold": 0.33,
                },
                {
                    "key": "cash_to_market_cap",
                    "label": "Cash / Market cap",
                    "numerator": "cash",
                    "denominator": "market_cap",
                    "max_threshold": 0.33,
                },
                {
                    "key": "current_assets_to_total_assets",
                    "label": "Current assets / Total assets",
                    "numerator": "current_assets",
                    "denominator": "total_assets",
                    "min_threshold": 0.10,
                },
            ]
        },
        "income_screen": {
            "max_non_core_income_ratio": 0.05,
        },
        "limitations": [
            "Business activity screening is only a placeholder based on sector and industry text.",
            "Detailed revenue segmentation is not yet implemented.",
            "Yahoo Finance data may be incomplete, delayed, unofficial, or missing for some tickers.",
            "SEC income screening is best-effort and only works when the filing exposes usable XBRL facts.",
            "The income screen now checks more SEC concept types, including interest, dividend, investment, and some non-operating income concepts.",
            "This app cannot reliably detect every form of non-compliant income from statements alone.",
            "If important values are missing, the app will return Insufficient data instead of guessing.",
        ],
    }
