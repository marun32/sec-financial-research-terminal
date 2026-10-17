import pandas as pd
import plotly.express as px
import streamlit as st

from sec_fin_rag.config import get_settings
from sec_fin_rag.ui import artifact_notice, issuers, page_setup

page_setup("Company Research")
universe = issuers()
selected = st.selectbox(
    "Company", universe, format_func=lambda item: f"{item.primary_ticker} — {item.name}"
)
path = get_settings().app_data_dir / "fundamentals.parquet"
if not path.exists():
    artifact_notice()
    st.stop()

facts = pd.read_parquet(path)
company = facts[facts["ticker"] == selected.primary_ticker].copy()
metrics = st.multiselect(
    "Metrics",
    sorted(company["metric"].unique()),
    default=[
        metric
        for metric in ("revenue", "net_income", "operating_cash_flow")
        if metric in set(company["metric"])
    ],
)
annual = company[company["form"].str.startswith("10-K") & company["metric"].isin(metrics)]
annual = annual.sort_values("end").drop_duplicates(["metric", "end"], keep="last")
if not annual.empty:
    figure = px.line(
        annual, x="end", y="value", color="metric", markers=True, title="SEC XBRL history"
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        annual[["metric", "value", "unit", "end", "filed", "form", "accession"]].sort_values(
            "end", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No normalized annual facts matched the selected metrics.")
