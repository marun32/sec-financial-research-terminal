import plotly.express as px
import streamlit as st

from sec_fin_rag.ui import artifact_notice, page_setup, scorecards

page_setup("Compare and Rank")
st.warning("Cross-sector scores are descriptive screening tools, not investment recommendations.")
frame = scorecards()
if frame.empty:
    artifact_notice()
    st.stop()

scorecard = st.selectbox("Scorecard", ["Balanced", "Growth", "Profitability", "Financial Strength"])
ranked = frame.sort_values(scorecard, ascending=False).copy()
ranked["Rank"] = range(1, len(ranked) + 1)
figure = px.bar(
    ranked,
    x="ticker",
    y=scorecard,
    color="sector",
    hover_data=["issuer", f"{scorecard}_coverage"],
    title=f"{scorecard} percentile score",
)
st.plotly_chart(figure, use_container_width=True)
st.dataframe(
    ranked[["Rank", "ticker", "issuer", "sector", scorecard, f"{scorecard}_coverage"]],
    use_container_width=True,
    hide_index=True,
)
with st.expander("Method and weights"):
    st.markdown(
        "Scores use winsorized cohort percentiles. Missing metric weights are redistributed across "
        "available metrics and coverage is displayed above. Balanced is the equal-weighted mean of "
        "Growth, Profitability, and Financial Strength."
    )
