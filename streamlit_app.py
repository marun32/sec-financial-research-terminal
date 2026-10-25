# ruff: noqa: E501
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from sec_fin_rag.market import market_snapshot
from sec_fin_rag.ui import (
    artifact_notice,
    issuers,
    market_ticker,
    metric_card,
    page_setup,
    scorecards,
    section_header,
)

page_setup(
    "Public-company research, grounded in filings",
    "Explore reported fundamentals, compare peers, and ask questions backed by SEC evidence.",
    eyebrow="FIN AI · MARKET DASHBOARD",
)

universe = issuers()
cards = scorecards()


@st.cache_data(ttl=300, show_spinner=False)
def cached_market_snapshot(tickers: tuple[str, ...]) -> list[dict[str, object]]:
    return market_snapshot(list(tickers))


market_rows: list[dict[str, object]] = []
market_error = ""
if os.getenv("FINRAG_DISABLE_MARKET_AUTOLOAD", "").lower() != "true":
    try:
        market_rows = cached_market_snapshot(tuple(item.primary_ticker for item in universe))
    except Exception as error:
        market_error = type(error).__name__

market_ticker(market_rows)
ticker_note, ticker_action = st.columns([5, 1])
with ticker_note:
    if market_rows:
        st.caption("Delayed quote context · yfinance · cached for five minutes")
    elif market_error:
        st.caption("Delayed quotes are temporarily unavailable; SEC research remains available.")
with ticker_action:
    if st.button("Refresh quotes", icon="🔄", width="stretch"):
        cached_market_snapshot.clear()
        st.rerun()

summary_cols = st.columns(4)
with summary_cols[0]:
    metric_card("Research universe", f"{len(universe)} companies", "S&P 500 Top 10 snapshot")
with summary_cols[1]:
    metric_card("SEC coverage", "3 × 10-K · 4 × 10-Q", "Per issuer, plus amendments")
with summary_cols[2]:
    latest_year = int(cards["fiscal_year"].max()) if not cards.empty else 2025
    metric_card("Latest fiscal year", str(latest_year), "Normalized SEC XBRL facts")
with summary_cols[3]:
    metric_card("Snapshot locked", "Sep 30, 2026", "Reproducible research universe")

section_header(
    "Latest reported fundamentals",
    "Comparable fiscal-year measures from SEC XBRL. Each percentage describes the company’s own reported performance—not a rating.",
)
if cards.empty:
    artifact_notice()
else:
    board = cards[
        [
            "ticker",
            "issuer",
            "sector",
            "period_end",
            "revenue_growth_yoy",
            "operating_margin",
        ]
    ].copy()
    board["revenue_growth_yoy"] = board["revenue_growth_yoy"] * 100
    board["operating_margin"] = board["operating_margin"] * 100
    board.columns = [
        "Ticker",
        "Company",
        "Sector",
        "Period ended",
        "Revenue change vs prior year",
        "Operating income ÷ revenue",
    ]
    st.dataframe(
        board.sort_values("Revenue change vs prior year", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "Revenue change vs prior year": st.column_config.NumberColumn(
                "Revenue growth",
                help="Change in annual revenue from the immediately preceding fiscal period.",
                format="%+.1f%%",
            ),
            "Operating income ÷ revenue": st.column_config.NumberColumn(
                "Operating margin",
                help="Operating income divided by revenue for the displayed annual period.",
                format="%.1f%%",
            ),
        },
    )

section_header("Start a research workflow", "Move from reported figures to filing evidence.")
workflow_cols = st.columns(3)
with workflow_cols[0]:
    st.page_link(
        "pages/1_Company_Intelligence.py", label="Explore one company", icon="🏢", width="stretch"
    )
    st.caption("Aligned annual KPIs, trends, and filing history.")
with workflow_cols[1]:
    st.page_link(
        "pages/2_Peer_Analysis.py", label="Compare selected peers", icon="⚖️", width="stretch"
    )
    st.caption("Transparent score drivers and underlying values.")
with workflow_cols[2]:
    st.page_link(
        "pages/3_Research_Assistant.py", label="Ask with SEC evidence", icon="🔎", width="stretch"
    )
    st.caption("Structured facts plus cited filing context.")

section_header("Coverage universe", "Companies included in the frozen September 2026 snapshot.")
universe_frame = pd.DataFrame(
    [
        {
            "Ticker": issuer.primary_ticker,
            "Company": issuer.name,
            "Sector": issuer.sector,
            "CIK": f"{issuer.cik:010d}",
        }
        for issuer in universe
    ]
)
st.dataframe(universe_frame, width="stretch", hide_index=True)
