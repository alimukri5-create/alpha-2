"""Streamlit app for the legacy screener and AlphaForge Phase 1 MVP."""

import streamlit as st

from alphaforge.config import DEFAULT_PERIOD, DEFAULT_TICKER, SUPPORTED_PERIODS
from alphaforge.data.fetch import get_price_history
from alphaforge.models.fusion import build_trade_map
from alphaforge.models.realized_vol import calculate_realized_volatility
from alphaforge.models.technicals import calculate_technical_structure
from data_fetcher import get_stock_data
from methodology import get_default_methodology
from screener import screen_stock
from utils import clean_ticker, format_number, format_percentage, get_status_label


st.set_page_config(page_title="AlphaForge")


def show_ratio_table(ratio_results: list[dict]) -> None:
    """Display ratio checks in a small beginner-friendly table."""
    if not ratio_results:
        st.info("No financial ratio checks were available.")
        return

    table_rows = []
    for item in ratio_results:
        table_rows.append(
            {
                "Ratio": item["label"],
                "Value": format_percentage(item.get("value")),
                "Threshold": item["threshold_label"],
                "Result": get_status_label(item["status"]),
                "Note": item["note"],
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def show_result(result: dict) -> None:
    """Render the screener result on the page."""
    company = result["company"]
    methodology = result["methodology"]
    business = result["business_screen"]
    financial = result["financial_screen"]
    income = result["income_screen"]

    st.subheader("Result")

    col1, col2 = st.columns(2)
    col1.metric("Company", company["company_name"])
    col2.metric("Ticker", company["ticker"])

    st.write(f"**Methodology:** {methodology['name']}")
    st.write(methodology["description"])

    verdict = result["final_verdict"]
    if verdict == "Compliant":
        st.success(f"Final verdict: {verdict}")
    elif verdict == "Non-compliant":
        st.error(f"Final verdict: {verdict}")
    else:
        st.warning(f"Final verdict: {verdict}")

    st.subheader("Screen Checks")
    st.write(f"**Business activity screen:** {get_status_label(business['status'])}")
    st.write(business["note"])

    st.write(f"**Financial ratio screen:** {get_status_label(financial['status'])}")
    st.write(financial["note"])

    st.write(f"**Income generation screen:** {get_status_label(income['status'])}")
    st.write(income["note"])

    st.subheader("Ratio Values")
    show_ratio_table(financial["ratio_results"])

    st.subheader("Income Screen Details")
    st.write(f"**Threshold:** {income['threshold_label']}")
    st.write(
        {
            "Selected possible non-compliant income fact": (
                income["selected_non_core_income_fact"]["label"]
                if income.get("selected_non_core_income_fact")
                else "Not found"
            ),
            "Selected fact category": (
                income["selected_non_core_income_fact"].get("category", "Unknown")
                if income.get("selected_non_core_income_fact")
                else "Not found"
            ),
            "Selected fact value": format_number(
                income["selected_non_core_income_fact"]["value"]
                if income.get("selected_non_core_income_fact")
                else None
            ),
            "Revenue fact": (
                income["revenue_fact"]["label"]
                if income.get("revenue_fact")
                else "Not found"
            ),
            "Revenue value": format_number(
                income["revenue_fact"]["value"]
                if income.get("revenue_fact")
                else None
            ),
            "Possible non-compliant income / Revenue": format_percentage(
                income.get("non_core_income_ratio")
            ),
        }
    )
    if income.get("non_core_income_facts"):
        st.write("**Other SEC candidate facts found:**")
        for item in income["non_core_income_facts"]:
            st.write(
                f"- {item['label']} ({item['category']}): {format_number(item['value'])}"
            )

    st.subheader("Plain-English Explanation")
    st.write(result["plain_english_explanation"])

    st.subheader("Confidence And Limitations")
    for note in result["limitations"]:
        st.write(f"- {note}")

    st.info(
        "This tool applies a stated screening methodology to prototype financial data. "
        "It is not a fatwa, religious ruling, or substitute for qualified scholarly advice."
    )

    with st.expander("Raw Data Used"):
        st.write(
            {
                "Sector": company.get("sector") or "Not available",
                "Industry": company.get("industry") or "Not available",
                "Market cap": format_number(company.get("market_cap")),
                "Total debt": format_number(company.get("total_debt")),
                "Cash": format_number(company.get("cash")),
                "Total assets": format_number(company.get("total_assets")),
                "Current assets": format_number(company.get("current_assets")),
            }
        )


def format_price(value: float | None) -> str:
    """Format a price-like float for display."""
    if value is None:
        return "Not available"
    return f"${value:,.2f}"


def format_vol(value: float | None) -> str:
    """Format volatility values as percentages."""
    if value is None:
        return "Not available"
    return f"{value:.1%}"


def show_alphaforge_screen() -> None:
    """Render the new AlphaForge Phase 1 screen."""
    st.title("AlphaForge Phase 1")
    st.write(
        "This MVP builds a disciplined probability + regime + execution map from daily price data. "
        "Phase 1 uses realized volatility and technical structure only."
    )

    col1, col2 = st.columns([2, 1])
    ticker_input = col1.text_input("Ticker symbol", value=DEFAULT_TICKER, key="alphaforge_ticker")
    period = col2.selectbox(
        "History period",
        SUPPORTED_PERIODS,
        index=SUPPORTED_PERIODS.index(DEFAULT_PERIOD),
    )

    if st.button("Run AlphaForge Analysis", type="primary"):
        with st.spinner("Fetching price history and building the trade map..."):
            fetch_result = get_price_history(ticker_input, period)

        if not fetch_result.ok or fetch_result.prices is None:
            st.error(fetch_result.error or "Could not fetch price history.")
            for note in fetch_result.notes:
                st.write(f"- {note}")
            return

        prices = fetch_result.prices
        realized_vol = calculate_realized_volatility(prices)
        technical = calculate_technical_structure(prices)
        trade_map = build_trade_map(realized_vol, technical)

        st.subheader(f"{fetch_result.ticker} Trade Map")

        header_col1, header_col2, header_col3 = st.columns(3)
        header_col1.metric("Current price", format_price(technical.current_price))
        header_col2.metric("Realized-vol regime", realized_vol.regime_label)
        header_col3.metric("Trend structure", technical.trend_label)

        st.subheader("Summary")
        st.write(trade_map.thesis_summary)
        st.write(f"**Confidence summary:** {trade_map.confidence_summary}")
        st.write(f"**Primary drivers:** {trade_map.driver_summary}")

        st.subheader("Realized-Vol Section")
        rv_col1, rv_col2, rv_col3, rv_col4 = st.columns(4)
        rv_col1.metric("HV 10d", format_vol(realized_vol.hv_10))
        rv_col2.metric("HV 20d", format_vol(realized_vol.hv_20))
        rv_col3.metric("HV 60d", format_vol(realized_vol.hv_60))
        rv_col4.metric("Short-horizon fair move", format_price(realized_vol.short_horizon_move))
        st.write(realized_vol.summary)
        st.write(f"**Distribution summary:** {trade_map.distribution_summary}")

        st.subheader("Technical Structure Section")
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        tech_col1.metric("Support", format_price(technical.support))
        tech_col2.metric("Resistance", format_price(technical.resistance))
        tech_col3.metric("ATR 14", format_price(technical.atr_14))
        tech_col4.metric("Breakout level", format_price(technical.breakout_level))
        st.write(technical.structure_summary)
        st.write(
            {
                "MA 20": format_price(technical.moving_averages["ma_20"]),
                "MA 50": format_price(technical.moving_averages["ma_50"]),
                "MA 100": format_price(technical.moving_averages["ma_100"]),
                "MA 200": format_price(technical.moving_averages["ma_200"]),
            }
        )

        st.subheader("Jump-Risk Section")
        st.info(
            "Jump and event-risk analysis is not implemented in Phase 1 yet. "
            "The current trade map therefore uses conservative stop placement."
        )

        st.subheader("Final Trade Map Section")
        st.write(f"**Starter-buy zone:** {trade_map.starter_buy_zone}")
        st.write(f"**Size-buy zone:** {trade_map.size_buy_zone}")
        st.write(f"**No-trade zone:** {trade_map.no_trade_zone}")
        st.write(f"**Breakout-confirmation zone:** {trade_map.breakout_confirmation_zone}")
        st.write(f"**Tactical stop:** {trade_map.tactical_stop}")
        st.write(f"**Structural stop:** {trade_map.structural_stop}")
        st.write(f"**Regime summary:** {trade_map.regime_summary}")

        with st.expander("Diagnostics"):
            st.write({"notes": fetch_result.notes, "fragility_notes": trade_map.fragility_notes})
            st.dataframe(prices.tail(20), use_container_width=True, hide_index=True)

    st.caption(
        "Phase 1 data source: Yahoo Finance price history via yfinance. "
        "No options-implied distribution, jump layer, or state-space regime model is included yet."
    )


def show_legacy_screener() -> None:
    """Render the original screener with minimal changes."""
    st.title("Shariah Stock Screener MVP")
    st.write(
        "Enter a US stock ticker to run a simple, transparent methodology-based "
        "Shariah screening check."
    )

    ticker_input = st.text_input("Ticker symbol", placeholder="Example: AAPL", key="legacy_ticker")
    run_screening = st.button("Run Screening")

    st.caption(
        "Prototype data sources: Yahoo Finance and SEC EDGAR. Some fields may be missing "
        "or incomplete."
    )

    if run_screening:
        ticker = clean_ticker(ticker_input)
        if not ticker:
            st.error("Please enter a ticker symbol before clicking Run Screening.")
            return

        methodology = get_default_methodology()

        with st.spinner("Fetching stock data and running the screening..."):
            stock_data = get_stock_data(ticker)

        if stock_data["status"] != "ok":
            st.error(stock_data["message"])
            if stock_data.get("limitations"):
                st.write("Known limitations:")
                for note in stock_data["limitations"]:
                    st.write(f"- {note}")
            return

        result = screen_stock(stock_data, methodology)
        show_result(result)

    st.subheader("Important Note")
    st.write(
        "This app is designed for learning and prototype screening only. It uses simple "
        "rules and limited data, especially for business activity screening."
    )


def main() -> None:
    """Build the Streamlit page."""
    tab1, tab2 = st.tabs(["AlphaForge Phase 1", "Legacy Screener"])
    with tab1:
        show_alphaforge_screen()
    with tab2:
        show_legacy_screener()


if __name__ == "__main__":
    main()
