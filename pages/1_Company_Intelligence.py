# ruff: noqa: E501
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from sec_fin_rag.config import get_settings
from sec_fin_rag.fact_periods import (
    PERIOD_LABELS,
    annual_fact_rows,
    quarter_fact_rows,
    with_fact_periods,
)
from sec_fin_rag.filing_catalog import sec_filing_index_url
from sec_fin_rag.ui import (
    artifact_notice,
    format_large,
    issuers,
    metric_card,
    metric_label,
    page_setup,
    pill,
    plotly_layout,
    scorecards,
    section_header,
)

page_setup(
    "Company Intelligence",
    "Trace reported performance, financial position, and filing history for a single issuer.",
    eyebrow="FIN AI · COMPANY RESEARCH",
)

universe = issuers()
selected = st.selectbox(
    "Research company",
    universe,
    format_func=lambda item: f"{item.primary_ticker}  ·  {item.name}",
)

path = get_settings().app_data_dir / "fundamentals.parquet"
if not path.exists():
    artifact_notice()
    st.stop()

facts = pd.read_parquet(path)
company = with_fact_periods(facts[facts["ticker"] == selected.primary_ticker].copy())
annual = annual_fact_rows(company[company["form"].str.startswith("10-K")])
annual = annual.sort_values(["end", "filed"]).drop_duplicates(["metric", "end"], keep="last")

header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown(
        f"### {selected.name} · `{selected.primary_ticker}`\n"
        + pill(selected.sector, accent=True)
        + pill(f"CIK {selected.cik:010d}")
        + pill(f"Snapshot {selected.snapshot_date}"),
        unsafe_allow_html=True,
    )
with header_right:
    st.link_button(
        "View SEC company page ↗",
        f"https://www.sec.gov/edgar/browse/?CIK={selected.cik:010d}",
        width="stretch",
    )

headline_metrics = ["revenue", "net_income", "operating_cash_flow", "assets"]
headline_facts = annual[annual["metric"].isin(headline_metrics)]
period_coverage = headline_facts.groupby("end")["metric"].nunique().sort_index()
complete_periods = period_coverage[period_coverage == len(headline_metrics)].index.tolist()
if complete_periods:
    headline_period = str(max(complete_periods))
elif not period_coverage.empty:
    best_coverage = int(period_coverage.max())
    headline_period = str(max(period_coverage[period_coverage == best_coverage].index))
else:
    headline_period = ""
aligned_headlines = headline_facts[headline_facts["end"] == headline_period]
latest_by_metric = {
    metric: rows.sort_values("filed").iloc[-1]
    for metric, rows in aligned_headlines.groupby("metric")
    if not rows.empty
}
metric_cols = st.columns(4)
for column, metric in zip(metric_cols, headline_metrics, strict=True):
    row = latest_by_metric.get(metric)
    with column:
        if row is None:
            metric_card(metric_label(metric), "—", "Not available")
        else:
            metric_card(
                metric_label(metric),
                format_large(float(row["value"]), str(row["unit"])),
                f"Period ended {headline_period} · filed {row['filed']}",
            )
st.caption(
    f"Headline values are aligned to one common annual period: {headline_period or 'not available'}."
)

overview_tab, trends_tab, filings_tab, raw_tab = st.tabs(
    ["Overview", "Financial trends", "Filing history", "Raw XBRL data"]
)

with overview_tab:
    section_header(
        "Financial profile",
        "Plain-language measures calculated from the same latest annual reporting period.",
    )
    company_score = scorecards()
    company_score = company_score[company_score["ticker"] == selected.primary_ticker]
    if not company_score.empty:
        row = company_score.iloc[0]
        definitions = {
            "revenue_growth_yoy": "Annual revenue change versus the immediately prior period.",
            "revenue_cagr_3y": "Annualized revenue growth over three years.",
            "operating_margin": "Operating income divided by revenue.",
            "net_margin": "Net income divided by revenue.",
            "return_on_assets": "Net income divided by total assets.",
            "fcf_margin": "Operating cash flow less capital expenditure, divided by revenue.",
            "cash_to_debt": "Cash and equivalents divided by reported debt.",
            "current_ratio": "Current assets divided by current liabilities.",
        }
        percentage_metrics = {
            "revenue_growth_yoy",
            "revenue_cagr_3y",
            "operating_margin",
            "net_margin",
            "return_on_assets",
            "fcf_margin",
        }
        available_profile = [
            metric for metric in definitions if metric in row.index and pd.notna(row[metric])
        ]

        def display_profile_value(metric: str) -> str:
            value = float(row[metric])
            return f"{value * 100:+.1f}%" if metric in percentage_metrics else f"{value:.2f}×"

        featured = available_profile[:4]
        if featured:
            profile_cards = st.columns(len(featured))
            for column, metric in zip(profile_cards, featured, strict=True):
                with column:
                    metric_card(
                        metric_label(metric), display_profile_value(metric), definitions[metric]
                    )
        profile = pd.DataFrame(
            [
                {
                    "Measure": metric_label(metric),
                    "Value": display_profile_value(metric),
                    "Meaning": definitions[metric],
                }
                for metric in available_profile
            ]
        )
        st.dataframe(
            profile,
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Derived from the annual period ending {row.get('period_end', headline_period)}. Unavailable measures are omitted rather than displayed as blanks."
        )
    section_header(
        "Continue with filing evidence", "Use a prepared question in the Research Assistant."
    )
    action_cols = st.columns(3)
    prompts = [
        f"Summarize the principal risks disclosed by {selected.name}.",
        f"Explain the major drivers of {selected.name}'s revenue and margins.",
        f"How has management's outlook for {selected.name} changed across filings?",
    ]
    for column, label, prompt in zip(
        action_cols,
        ["Analyze risks", "Explain performance", "Review outlook"],
        prompts,
        strict=True,
    ):
        with column:
            st.page_link(
                "pages/3_Research_Assistant.py",
                label=label,
                icon="🔎",
                query_params={"company": selected.primary_ticker, "prompt": prompt},
                width="stretch",
            )

with trends_tab:
    controls = st.columns([2, 1])
    available_metrics = sorted(company["metric"].dropna().unique())
    with controls[0]:
        trend_metric = st.selectbox(
            "Metric",
            available_metrics,
            format_func=metric_label,
            index=available_metrics.index("revenue") if "revenue" in available_metrics else 0,
        )
    with controls[1]:
        period = st.segmented_control("Periodicity", ["Annual", "Quarterly"], default="Annual")
    form_prefix = "10-K" if period == "Annual" else "10-Q"
    trend = company[
        company["form"].str.startswith(form_prefix) & (company["metric"] == trend_metric)
    ].copy()
    quarterly_basis = "3-month quarter"
    if period == "Annual":
        trend = annual_fact_rows(trend)
    elif trend["start"].notna().any():
        quarterly_basis = st.segmented_control(
            "Quarterly basis", ["3-month quarter", "Year to date"], default="3-month quarter"
        )
        if quarterly_basis == "Year to date":
            trend = trend[
                trend["period_type"].isin(["six_months", "nine_months"])
                | ((trend["period_type"] == "three_months") & (trend["fiscal_period"] == "Q1"))
            ]
        else:
            trend = quarter_fact_rows(trend)
    trend = trend.sort_values(["end", "duration_days", "filed"]).drop_duplicates("end", keep="last")
    if trend.empty:
        st.info("No normalized facts match this selection.")
    else:
        figure = px.line(
            trend,
            x="end",
            y="value",
            markers=True,
            labels={"end": "Period end", "value": metric_label(trend_metric)},
            title=f"{metric_label(trend_metric)} · {period.lower()} history",
        )
        figure.update_traces(line_color="#38bdf8", marker_color="#2dd4bf", line_width=3)
        st.plotly_chart(plotly_layout(figure), width="stretch")
        if period == "Annual":
            st.caption(
                "Annual chart: full-year duration facts and period-end balances. Values below "
                "are rounded for display; exact figures are in Raw XBRL data. Source report end "
                "shows whether the linked filing covers that same period."
            )
        elif quarterly_basis == "Year to date":
            st.caption(
                "Year-to-date view: six- and nine-month cumulative facts; Q1 is three months. "
                "These are not standalone quarterly amounts."
            )
        else:
            st.caption(
                "Quarterly view: standalone three-month facts or balances at period end. "
                "Use Year to date for cumulative SEC durations when available."
            )
        if (trend["source_report_end"].notna() & ~trend["source_period_match"]).any():
            st.warning(
                "Some older facts are available only as comparisons in later filings. "
                "Check Source report end before treating a link as the original-period filing."
            )
        if trend["source_report_end"].isna().any():
            st.caption(
                "A blank Source report end means historical SEC submission metadata was not "
                "cached; the earliest available filing is linked but its report period is unverified."
            )
        display = trend[
            [
                "start",
                "end",
                "period_type",
                "value",
                "unit",
                "filed",
                "form",
                "accession",
                "source_report_end",
            ]
        ].copy()
        display["SEC filing"] = display["accession"].map(
            lambda accession: sec_filing_index_url(selected.cik, accession)
        )
        display = display.drop(columns="accession")
        display["period_type"] = display["period_type"].map(PERIOD_LABELS)
        display["value"] = display.apply(
            lambda row: (
                f"${float(row['value']):,.2f}/share"
                if row["unit"] == "USD/shares"
                else format_large(float(row["value"]), str(row["unit"]))
            ),
            axis=1,
        )
        st.dataframe(
            display.sort_values("end", ascending=False).rename(
                columns={
                    "start": "Period start",
                    "end": "Period end",
                    "value": "Value",
                    "unit": "Unit",
                    "period_type": "Fact duration",
                    "filed": "Source filing date",
                    "form": "Form",
                    "source_report_end": "Source report end",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"SEC filing": st.column_config.LinkColumn(display_text="View filing ↗")},
        )

with filings_tab:
    filing_catalog_path = get_settings().app_data_dir / "filings.parquet"
    if not filing_catalog_path.exists():
        st.info("The filing catalog is unavailable. Run `finrag build-artifacts` to create it.")
    else:
        filings = pd.read_parquet(filing_catalog_path)
        filings = filings[filings["ticker"] == selected.primary_ticker].copy()
        filings = filings[
            [
                "form",
                "filing_date",
                "report_date",
                "fiscal_year",
                "fiscal_period",
                "accession",
                "source_url",
            ]
        ].sort_values("filing_date", ascending=False)
        filings = filings.rename(
            columns={
                "form": "Form",
                "filing_date": "Filed",
                "report_date": "Period end",
                "fiscal_year": "Fiscal year",
                "fiscal_period": "Fiscal period",
                "accession": "Accession",
                "source_url": "SEC filing",
            }
        )
        st.dataframe(
            filings,
            width="stretch",
            hide_index=True,
            column_config={
                "SEC filing": st.column_config.LinkColumn(display_text="Open SEC filing ↗")
            },
        )

with raw_tab:
    st.caption(
        "Detailed Company Facts records for validation and export. Fiscal year/period are labels "
        "of the source filing; start/end and fact duration describe each reported value."
    )
    raw_metrics = st.multiselect(
        "Filter metrics", available_metrics, format_func=metric_label, default=[]
    )
    raw = company if not raw_metrics else company[company["metric"].isin(raw_metrics)]
    st.dataframe(raw.sort_values("filed", ascending=False), width="stretch", hide_index=True)
    st.download_button(
        "Download selected rows as CSV",
        raw.to_csv(index=False),
        file_name=f"{selected.primary_ticker.lower()}_sec_facts.csv",
        mime="text/csv",
    )
