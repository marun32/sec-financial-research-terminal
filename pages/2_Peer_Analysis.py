# ruff: noqa: E501
from __future__ import annotations

import plotly.express as px
import streamlit as st

from sec_fin_rag.ui import (
    artifact_notice,
    metric_label,
    page_setup,
    plotly_layout,
    scorecards,
    section_header,
)

page_setup(
    "Peer Analysis",
    "Compare financial characteristics transparently—composite scores, underlying metrics, and data coverage together.",
    eyebrow="FIN AI · COMPARATIVE RESEARCH",
)

frame = scorecards()
if frame.empty:
    artifact_notice()
    st.stop()

SCORECARDS = {
    "Balanced": {
        "description": "Equal-weighted view of Growth, Profitability, and Financial Strength.",
        "metrics": ["Growth", "Profitability", "Financial Strength"],
        "weights": "33⅓% each",
    },
    "Growth": {
        "description": "Measures reported expansion in revenue and earnings.",
        "metrics": ["revenue_cagr_3y", "revenue_growth_yoy", "eps_growth"],
        "weights": "Revenue CAGR 40% · YoY revenue 30% · EPS growth 30%",
    },
    "Profitability": {
        "description": "Measures margins and the efficiency of the asset base.",
        "metrics": ["operating_margin", "net_margin", "return_on_assets", "fcf_margin"],
        "weights": "Operating margin 35% · Net margin 25% · ROA 20% · FCF margin 20%",
    },
    "Financial Strength": {
        "description": "Measures liquidity, leverage, and the ability to service obligations.",
        "metrics": ["cash_to_debt", "inverse_debt_to_assets", "current_ratio", "interest_coverage"],
        "weights": "Cash/debt 30% · Inverse debt/assets 25% · Current ratio 25% · Interest coverage 20%",
    },
}

control_cols = st.columns([1.1, 2.4])
with control_cols[0]:
    scorecard = st.selectbox("Research lens", list(SCORECARDS))
with control_cols[1]:
    available = frame["ticker"].tolist()
    selected = st.multiselect(
        "Peer group",
        available,
        default=available[:5],
        format_func=lambda ticker: (
            f"{ticker} · {frame.loc[frame['ticker'] == ticker, 'issuer'].iloc[0]}"
        ),
    )

if len(selected) < 2:
    st.info("Select at least two companies to build a peer comparison.")
    st.stop()

peers = frame[frame["ticker"].isin(selected)].copy()
definition = SCORECARDS[scorecard]
st.markdown(f"#### {scorecard}\n{definition['description']}")
st.caption(f"Weights: {definition['weights']}")

if peers["sector"].nunique() > 1:
    st.warning(
        "This peer group spans multiple sectors. Accounting structures, capital intensity, and normal margins differ; interpret ranks as screening context rather than direct valuation conclusions."
    )
else:
    st.success("Selected companies are in the same sector, improving operational comparability.")

ranked = peers.sort_values(scorecard, ascending=True).copy()
ranked["Rank"] = ranked[scorecard].rank(ascending=False, method="min").astype(int)
figure = px.bar(
    ranked,
    x=scorecard,
    y="ticker",
    orientation="h",
    color=scorecard,
    color_continuous_scale=[[0, "#17304a"], [0.5, "#1783ad"], [1, "#2dd4bf"]],
    range_color=[0, 100],
    hover_data={"issuer": True, "sector": True, f"{scorecard}_coverage": ":.0%"},
    labels={scorecard: "Cohort percentile", "ticker": ""},
    title=f"{scorecard} score · selected peer group",
)
figure.update_layout(coloraxis_showscale=False)
st.plotly_chart(plotly_layout(figure, height=max(340, len(peers) * 55)), width="stretch")

section_header(
    "What drives the ranking",
    "Reported or derived values sit beside the composite percentile and coverage. Missing weights are redistributed across available metrics.",
)
detail_columns = [
    "Rank",
    "ticker",
    "issuer",
    "sector",
    *definition["metrics"],
    scorecard,
    f"{scorecard}_coverage",
]
detail = ranked[detail_columns].sort_values("Rank").copy()
detail = detail.rename(
    columns={
        "ticker": "Ticker",
        "issuer": "Company",
        "sector": "Sector",
        scorecard: f"{scorecard} score",
        f"{scorecard}_coverage": "Coverage",
        **{metric: metric_label(metric) for metric in definition["metrics"]},
    }
)
config: dict[str, object] = {
    f"{scorecard} score": st.column_config.ProgressColumn(
        min_value=0, max_value=100, format="%.1f"
    ),
    "Coverage": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0f%%"),
}
st.dataframe(detail, width="stretch", hide_index=True, column_config=config)

percentile_columns = [
    f"{metric}_percentile"
    for metric in definition["metrics"]
    if f"{metric}_percentile" in peers.columns
]
if percentile_columns:
    section_header(
        "Metric percentile map",
        "Higher values indicate a stronger relative position within the full ten-company snapshot cohort.",
    )
    heatmap = peers.set_index("ticker")[percentile_columns]
    heatmap.columns = [
        metric_label(column.removesuffix("_percentile")) for column in heatmap.columns
    ]
    heatmap_figure = px.imshow(
        heatmap,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale=[[0, "#101d30"], [0.5, "#176b8b"], [1, "#2dd4bf"]],
        zmin=0,
        zmax=100,
        labels={"color": "Percentile"},
    )
    st.plotly_chart(plotly_layout(heatmap_figure, height=330), width="stretch")

with st.expander("How to interpret these scores"):
    st.markdown(
        "Scores are winsorized percentile composites within this fixed ten-company universe. "
        "They are descriptive screening tools, not forecasts, valuations, or investment ratings. "
        "Coverage shows how much of the intended metric weight was available before redistribution."
    )
