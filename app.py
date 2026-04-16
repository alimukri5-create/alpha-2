"""Streamlit app for the legacy screener and AlphaForge Phase 1 MVP."""

from datetime import datetime

import pandas as pd
import streamlit as st

from alphaforge.config import DEFAULT_PERIOD, DEFAULT_TICKER, SUPPORTED_PERIODS
from alphaforge.data.fetch import get_price_history
from alphaforge.models.fusion import build_trade_map
from alphaforge.models.autonomous_vectorizer import (
    analyze_ticker,
    analyze_tickers,
    export_autonomous_results_json,
    parse_ticker_csv,
    parse_ticker_text,
)
from alphaforge.models.momentum_vectorizer import (
    MomentumSignalVectorizer,
    export_signals_for_claude,
    export_signals_json,
)
from alphaforge.models.realized_vol import calculate_realized_volatility
from alphaforge.models.technicals import calculate_technical_structure
from data_fetcher import get_stock_data
from methodology import get_default_methodology
from screener import screen_stock
from utils import clean_ticker, format_number, format_percentage, get_status_label


st.set_page_config(page_title="AlphaForge")

VECTORIZER = MomentumSignalVectorizer()


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


def format_tier_label(signal: dict) -> str:
    """Format a concise tier label for the summary table."""
    return signal["local_tier"]


def ensure_signal_state() -> list[dict]:
    """Initialize session storage for manual vectorized signals."""
    if "vectorized_signals" not in st.session_state:
        st.session_state["vectorized_signals"] = []
    return st.session_state["vectorized_signals"]


def lookup_shariah_context(ticker: str) -> dict:
    """Run the existing screener stack for optional context lookup."""
    cleaned_ticker = clean_ticker(ticker)
    if not cleaned_ticker:
        return {"status": "error", "message": "Enter a ticker before running the lookup."}

    stock_data = get_stock_data(cleaned_ticker)
    if stock_data["status"] != "ok":
        return {
            "status": "error",
            "message": stock_data["message"],
            "limitations": stock_data.get("limitations", []),
        }

    result = screen_stock(stock_data, get_default_methodology())
    debt_to_market_cap = None
    for ratio in result["financial_screen"]["ratio_results"]:
        if ratio["key"] == "debt_to_market_cap":
            debt_to_market_cap = ratio.get("value")
            break

    return {
        "status": "ok",
        "ticker": cleaned_ticker,
        "final_verdict": result["final_verdict"],
        "business_status": result["business_screen"]["status"],
        "financial_status": result["financial_screen"]["status"],
        "income_status": result["income_screen"]["status"],
        "debt_to_market_cap": debt_to_market_cap,
        "limitations": result.get("limitations", []),
        "plain_english_explanation": result["plain_english_explanation"],
    }


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


def show_momentum_vectorizer() -> None:
    """Render the manual momentum vectorizer workflow."""
    st.title("Momentum Vectorizer")
    st.write(
        "Turn manual momentum observations into structured JSON, a local edge tier, "
        "and a copy-ready Claude review prompt."
    )

    signals_log = ensure_signal_state()
    lookup_ticker = st.text_input("Optional lookup ticker", value="", key="vectorizer_lookup_ticker")

    lookup_state = st.session_state.get("vectorizer_lookup_result")
    if lookup_state:
        if lookup_state["status"] == "ok":
            st.info(
                f"Optional screener lookup for {lookup_state['ticker']}: "
                f"{lookup_state['final_verdict']} "
                f"(business {get_status_label(lookup_state['business_status'])}, "
                f"financial {get_status_label(lookup_state['financial_status'])}, "
                f"income {get_status_label(lookup_state['income_status'])})."
            )
            if lookup_state.get("debt_to_market_cap") is not None:
                st.caption(
                    f"Fetched debt / market cap: {format_percentage(lookup_state['debt_to_market_cap'])}"
                )
            for note in lookup_state.get("limitations", []):
                st.write(f"- {note}")
        else:
            st.warning(lookup_state["message"])
            for note in lookup_state.get("limitations", []):
                st.write(f"- {note}")

    lookup_col1, lookup_col2 = st.columns([1, 3])
    if lookup_col1.button("Run Optional Shariah Lookup"):
        with st.spinner("Fetching existing screener context..."):
            st.session_state["vectorizer_lookup_result"] = lookup_shariah_context(lookup_ticker)
        st.rerun()

    if lookup_col2.button("Clear Lookup Context"):
        st.session_state.pop("vectorizer_lookup_result", None)
        st.rerun()

    with st.form("momentum_vectorizer_form", clear_on_submit=False):
        ticker_col, price_col, debt_col = st.columns(3)
        ticker = ticker_col.text_input("Ticker", value="")
        current_price = price_col.number_input("Current price", min_value=0.0, value=10.0, step=0.01)
        default_debt_ratio = 0.2
        if lookup_state and lookup_state.get("status") == "ok" and lookup_state.get("debt_to_market_cap") is not None:
            default_debt_ratio = float(lookup_state["debt_to_market_cap"])
        debt_to_mcap = debt_col.number_input(
            "Debt / market cap",
            min_value=0.0,
            max_value=5.0,
            value=float(default_debt_ratio),
            step=0.01,
            help="Manual-first input. Optional screener lookup can suggest a value.",
        )

        signal_col, win_col, hold_col = st.columns(3)
        entry_signal_strength = signal_col.slider("Entry signal strength", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
        prior_pattern_win_rate = win_col.slider("Prior pattern win rate", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
        swing_target_days = hold_col.number_input("Target hold (days)", min_value=1, max_value=60, value=5, step=1)

        catalyst_col1, catalyst_col2, catalyst_col3 = st.columns(3)
        catalyst_type = catalyst_col1.selectbox(
            "Catalyst type",
            ["earnings", "macro", "sector", "supply_shock", "insider_activity", "none"],
        )
        catalyst_magnitude = catalyst_col2.selectbox("Catalyst magnitude", ["major", "moderate", "minor"])
        catalyst_timing = catalyst_col3.selectbox("Catalyst timing", ["today", "this_week", "next_week", "uncertain"])

        action_col1, action_col2, action_col3 = st.columns(3)
        price_action_timeframe = action_col1.selectbox("Price-action timeframe", ["intraday", "1d", "multiday", "weekly"])
        vol_vs_20day_avg = action_col2.number_input("Volume vs 20-day avg", min_value=0.0, value=1.2, step=0.1)
        price_momentum_5d = action_col3.number_input("5-day momentum (%)", value=0.0, step=0.1)

        confluence_text = st.text_area(
            "Confluence factors",
            value="",
            help="Enter one factor per line, or separate them with commas.",
        )
        notes = st.text_area("Notes", value="")

        add_signal = st.form_submit_button("Add Signal", type="primary")

    if add_signal:
        cleaned_ticker = clean_ticker(ticker)
        if not cleaned_ticker:
            st.error("Please enter a ticker before adding a signal.")
        else:
            confluence_factors = [
                item.strip()
                for chunk in confluence_text.splitlines()
                for item in chunk.split(",")
                if item.strip()
            ]
            signal = VECTORIZER.vectorize_signal(
                ticker=cleaned_ticker,
                current_price=current_price,
                entry_signal_strength=entry_signal_strength,
                confluence_factors=confluence_factors,
                catalyst_type=catalyst_type,
                catalyst_magnitude=catalyst_magnitude,
                catalyst_timing=catalyst_timing,
                price_action_timeframe=price_action_timeframe,
                vol_vs_20day_avg=vol_vs_20day_avg,
                price_momentum_5d=price_momentum_5d,
                debt_to_mcap=debt_to_mcap,
                swing_target_days=swing_target_days,
                prior_pattern_win_rate=prior_pattern_win_rate,
                notes=notes,
                shariah_lookup=lookup_state if lookup_state and lookup_state.get("status") == "ok" else None,
            )
            signals_log.append(signal.to_dict())
            st.success(f"Added {cleaned_ticker} as {signal.local_tier}.")

    st.subheader("Signals")
    if not signals_log:
        st.info("No signals added yet. Fill the form, then add your first setup.")
        return

    summary_rows = []
    for signal in signals_log:
        summary_rows.append(
            {
                "Ticker": signal["ticker"],
                "Price": signal["price"],
                "Signal Strength": signal["signal_strength"],
                "Confluence": signal["confluence_count"],
                "Catalyst": signal["catalyst"]["type"],
                "Conviction": signal["scores"]["conviction"],
                "Timeframe": signal["scores"]["timeframe_alignment"],
                "Volume": signal["scores"]["volume_confirmation"],
                "Catalyst Score": signal["scores"]["catalyst_strength"],
                "Composite": signal["scores"]["composite_quality"],
                "Tier": format_tier_label(signal),
                "Shariah": "Pass" if signal["shariah_compliance"]["passed"] else "Fail",
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Clear Signals"):
        st.session_state["vectorized_signals"] = []
        st.rerun()

    export_json = export_signals_json(signals_log)
    filename = f"momentum_signals_{datetime.now().strftime('%Y%m%d')}.json"
    action_col2.download_button(
        "Download JSON",
        data=export_json,
        file_name=filename,
        mime="application/json",
    )

    st.subheader("Claude Prompt")
    st.code(export_signals_for_claude(signals_log), language="text")

    st.subheader("Raw JSON")
    st.code(export_json, language="json")

    st.caption(
        "This workflow is manual-first and export-focused. The optional Shariah lookup uses the existing "
        "prototype screener stack and may be incomplete or rate-limited."
    )


@st.cache_data(ttl=300, show_spinner=False)
def run_autonomous_single(ticker: str) -> dict:
    """Run the autonomous engine for one ticker with short-lived caching."""
    return analyze_ticker(ticker).to_dict()


@st.cache_data(ttl=300, show_spinner=False)
def run_autonomous_batch(tickers: tuple[str, ...]) -> list[dict]:
    """Run the autonomous engine for a batch of tickers with short-lived caching."""
    return [result.to_dict() for result in analyze_tickers(list(tickers))]


def _format_signal_badge(value: str) -> str:
    """Render a concise status label."""
    mapping = {"BUY": "BUY", "WATCH": "WATCH", "SKIP": "SKIP", "DISQUALIFIED": "DISQUALIFIED"}
    return mapping.get(value, value)


def _autonomous_summary_rows(results: list[dict]) -> list[dict]:
    """Flatten result dictionaries into summary table rows."""
    rows = []
    for item in results:
        scores = item["scores"]
        shariah = item["shariah"]
        rows.append(
            {
                "Ticker": item["ticker"],
                "Price": item["latest_price"],
                "Momentum%": item["price_change_5d_pct"],
                "Volume": item["technical_metrics"].get("volume", {}).get("volume_ratio_20d"),
                "BBW Comp": scores["bbw"],
                "RS": item["relative_strength_vs_spy"],
                "Regime": item["regime"],
                "Confluence": scores["confluence"],
                "Catalyst": item["catalyst"]["catalyst_type"],
                "Shariah": shariah["status"],
                "Composite": scores["composite"],
                "Tier": item["tier"],
                "Signal": _format_signal_badge(item["trade_signal"]),
                "Status": item["status"],
            }
        )
    return rows


def show_autonomous_result_card(result: dict) -> None:
    """Render one detailed autonomous analysis card."""
    scores = result["scores"]
    metrics = result["technical_metrics"]
    shariah = result["shariah"]
    headline = f"{result['ticker']} | {result['tier']} | {result['trade_signal']} | Composite {scores['composite']:.2f}"
    with st.expander(headline):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"${result['latest_price']:.2f}" if result["latest_price"] is not None else "N/A")
        col2.metric("5d Momentum", format_percentage(result["price_change_5d_pct"]))
        col3.metric("Volume Ratio", f"{metrics.get('volume', {}).get('volume_ratio_20d', 0):.2f}x" if metrics.get("volume", {}).get("volume_ratio_20d") is not None else "N/A")
        col4.metric("RS vs SPY", format_percentage(result["relative_strength_vs_spy"]))
        st.caption(f"Analyzed at PKT: {result['analyzed_at_pk_time']}")

        st.write(
            f"**Shariah:** {shariah['status']} | **Regime:** {result['regime']} | "
            f"**Volatility:** {result['volatility_regime']} | **Catalyst:** {result['catalyst']['summary']}"
        )
        if result.get("error"):
            st.error(result["error"])

        if result["flags"]:
            for flag in result["flags"]:
                st.warning(flag)

        st.write(f"Momentum Score: `{scores['momentum']:.2f}`")
        st.write(result["narratives"].get("momentum", ""))
        st.write(f"Volume Score: `{scores['volume']:.2f}`")
        st.write(result["narratives"].get("volume", ""))
        st.write(f"Confluence Score: `{scores['confluence']:.2f}`")
        st.write(result["narratives"].get("confluence", ""))
        st.write(f"BBW Score: `{scores['bbw']:.2f}`")
        st.write(result["narratives"].get("bbw", ""))
        st.write(f"Catalyst Score: `{scores['catalyst']:.2f}`")
        st.write(result["narratives"].get("catalyst", ""))
        st.write(f"Signal Quality: `{scores['significance']:.2f}`")
        st.write(result["narratives"].get("significance", ""))
        st.write(f"Shariah Note: {result['narratives'].get('shariah', '')}")

        with st.expander("Raw Technical Metrics"):
            st.json(metrics)


def show_autonomous_vectorizer() -> None:
    """Render the autonomous momentum vectorizer workflow."""
    st.title("Autonomous Vectorizer")
    st.write(
        "Enter only a ticker, or paste/upload a batch list. The app automatically fetches "
        "daily and intraday data, scores the setup, runs Shariah screening, and exports JSON."
    )
    st.caption("All timestamps and market context are shown in PKT (Asia/Karachi).")

    single_ticker = st.text_input("Single ticker", placeholder="Example: UUUU", key="autonomous_single_ticker")
    batch_text = st.text_area(
        "Batch tickers",
        placeholder="UUUU\nAXTI\nRKLB",
        help="Paste tickers separated by new lines, commas, or semicolons.",
        key="autonomous_batch_text",
    )
    batch_file = st.file_uploader("CSV upload", type=["csv"], key="autonomous_batch_csv")

    run_single = st.button("Run Autonomous Analysis", type="primary")
    run_batch = st.button("Run Batch Vectorization")

    if "autonomous_results" not in st.session_state:
        st.session_state["autonomous_results"] = []
    results: list[dict] = st.session_state["autonomous_results"]
    if run_single:
        ticker = clean_ticker(single_ticker)
        if not ticker:
            st.error("Please enter a ticker symbol.")
        else:
            with st.spinner("Running autonomous analysis..."):
                results = [run_autonomous_single(ticker)]
            st.session_state["autonomous_results"] = results

    if run_batch:
        batch_tickers = parse_ticker_text(batch_text)
        if batch_file is not None:
            batch_tickers.extend(parse_ticker_csv(batch_file.getvalue()))
        normalized = tuple(dict.fromkeys(batch_tickers))
        if not normalized:
            st.error("Please paste tickers or upload a CSV before running batch mode.")
        else:
            with st.spinner("Vectorizing the batch in parallel..."):
                results = run_autonomous_batch(normalized)
            st.session_state["autonomous_results"] = results

    if not results:
        st.info("No autonomous results yet. Run a single ticker or a batch to populate the table.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    tier_one_only = filter_col1.checkbox("Show Tier 1 only", value=False)
    bullish_only = filter_col2.checkbox("Show bullish regimes only", value=False)
    shariah_only = filter_col3.checkbox("Show Shariah pass only", value=False)

    filtered_results = []
    for item in results:
        if tier_one_only and item["tier"] != "Tier 1":
            continue
        if bullish_only and item["regime"] != "uptrend":
            continue
        if shariah_only and item["shariah"]["status"] != "PASS":
            continue
        filtered_results.append(item)

    summary = pd.DataFrame(_autonomous_summary_rows(filtered_results))
    if not summary.empty:
        st.subheader("Session Table")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.warning("The current filters removed all rows.")

    export_json = export_autonomous_results_json(filtered_results)
    filename = f"autonomous_vectorizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button("Download JSON", data=export_json, file_name=filename, mime="application/json")

    st.subheader("Detailed Analysis")
    for result in filtered_results:
        show_autonomous_result_card(result)


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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Autonomous Vectorizer", "AlphaForge Phase 1", "Legacy Screener", "Momentum Vectorizer"]
    )
    with tab1:
        show_autonomous_vectorizer()
    with tab2:
        show_alphaforge_screen()
    with tab3:
        show_legacy_screener()
    with tab4:
        show_momentum_vectorizer()


if __name__ == "__main__":
    main()
