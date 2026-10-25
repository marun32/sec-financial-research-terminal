# ruff: noqa: E501
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from sec_fin_rag.config import get_settings
from sec_fin_rag.ui import issuers, metric_card, page_setup, section_header

page_setup(
    "About & Data",
    "Understand the research universe, evidence sources, scoring approach, and important limitations.",
    eyebrow="FIN AI · TRANSPARENCY",
)

universe = issuers()
summary_cols = st.columns(4)
with summary_cols[0]:
    metric_card("Companies", str(len(universe)), "Frozen index snapshot")
with summary_cols[1]:
    metric_card("Annual filings", "Up to 30", "Three 10-Ks per issuer")
with summary_cols[2]:
    metric_card("Quarterly filings", "Up to 40", "Four 10-Qs per issuer")
with summary_cols[3]:
    metric_card("Snapshot date", "Sep 30, 2026", "Point-in-time reproducibility")

overview_tab, sources_tab, scoring_tab, limits_tab = st.tabs(
    ["Product scope", "Data sources", "Scorecards", "Limitations"]
)

with overview_tab:
    section_header("What Fin AI does")
    st.markdown(
        "Fin AI combines SEC filing narratives, structured XBRL fundamentals, and optional "
        "delayed market context. It supports company research, transparent peer screening, and "
        "questions answered against retrieved evidence. It does not execute trades or provide "
        "personalized investment recommendations."
    )
    section_header("Research universe")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ticker": item.primary_ticker,
                    "Company": item.name,
                    "Sector": item.sector,
                    "CIK": f"{item.cik:010d}",
                }
                for item in universe
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "The universe is intentionally frozen to the September 30, 2026 S&P 500 Top 10 Index snapshot so experiments and comparisons remain reproducible."
    )

with sources_tab:
    section_header("Primary evidence")
    st.markdown(
        """
        - **SEC EDGAR filings:** contextual paragraphs and tables from recent 10-K and 10-Q filings.
        - **SEC Company Facts:** normalized XBRL history used for trends, calculations, and charts.
        - **Yahoo Finance:** optional delayed quote context, clearly separated from the frozen SEC dataset.
        """
    )
    section_header("Document coverage")
    st.markdown(
        "For each issuer, ingestion targets the three latest 10-Ks and four latest 10-Qs available "
        "on or before the snapshot date. Amendments are retained as supplemental evidence. Filing "
        "answers link back to the original SEC source."
    )
    section_header("How answers are produced")
    st.markdown(
        "Questions are routed to filing retrieval, XBRL fundamentals, delayed market context, or a "
        "combination. Filing candidates use dense and BM25 sparse retrieval, local reranking, and "
        "a grounded generation step. Every accepted factual answer must contain valid references."
    )

with scoring_tab:
    section_header("Transparent screening lenses")
    scoring = pd.DataFrame(
        [
            {
                "Lens": "Growth",
                "Metrics and weights": "3-year revenue CAGR 40%; YoY revenue 30%; EPS growth 30%",
            },
            {
                "Lens": "Profitability",
                "Metrics and weights": "Operating margin 35%; net margin 25%; ROA 20%; FCF margin 20%",
            },
            {
                "Lens": "Financial Strength",
                "Metrics and weights": "Cash/debt 30%; inverse debt/assets 25%; current ratio 25%; interest coverage 20%",
            },
            {
                "Lens": "Balanced",
                "Metrics and weights": "Equal-weighted mean of Growth, Profitability, and Financial Strength",
            },
        ]
    )
    st.dataframe(scoring, width="stretch", hide_index=True)
    st.markdown(
        "Metrics are winsorized and converted to percentiles within the ten-company cohort. When a "
        "metric is missing, its weight is redistributed across available metrics and the resulting "
        "coverage is displayed. Scores are descriptive—not ratings or forecasts."
    )

with limits_tab:
    section_header("Important interpretation limits")
    st.markdown(
        """
        - Companies across sectors have different economics and accounting profiles.
        - XBRL concepts and reporting periods can vary between issuers.
        - Quarterly facts may be cumulative year-to-date values rather than standalone quarters.
        - Retrieval can miss relevant passages; absence in retrieved evidence is not proof of nondisclosure.
        - Yahoo Finance values may be delayed, incomplete, or temporarily unavailable.
        - Generated explanations should be verified against the linked filing before consequential use.
        """
    )

with st.expander("Technical validation results"):
    report_path = get_settings().reports_dir / "retrieval_evaluation.json"
    deployment_path = get_settings().app_data_dir / "evaluation_summary.json"
    path = report_path if report_path.exists() else deployment_path
    if not path.exists():
        st.caption("No evaluation summary is available in this environment.")
    else:
        report = json.loads(path.read_text())
        st.caption(
            f"Evaluation snapshot · recommended configuration: {report.get('recommended_configuration', 'not specified')}"
        )
        generation = report.get("generation", {})
        if generation:
            rows = [
                {"Measure": key.replace("_", " ").title(), "Score": value}
                for key, value in generation.items()
            ]
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.1%"
                    )
                },
            )
        st.caption(
            "These engineering measures validate retrieval and grounded generation. They are not company-performance indicators."
        )
