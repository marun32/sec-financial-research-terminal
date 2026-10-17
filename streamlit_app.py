import pandas as pd
import streamlit as st

from sec_fin_rag.market import market_snapshot
from sec_fin_rag.ui import issuers, page_setup

page_setup("Market Overview")

st.markdown(
    "A focused research terminal for the ten companies in the **S&P 500 Top 10 Index snapshot "
    "dated September 30, 2026**. Use the sidebar to open company research, rankings, chat, "
    "and evaluation."
)

universe = issuers()
st.subheader("Issuer universe")
cols = st.columns(5)
for index, issuer in enumerate(universe):
    with cols[index % 5]:
        st.metric(issuer.primary_ticker, issuer.name.split(" ")[0])
        st.caption(issuer.sector)

st.subheader("Delayed market snapshot")
if st.button("Load current snapshot", type="primary"):
    with st.spinner("Retrieving delayed market data…"):
        try:
            data = pd.DataFrame(market_snapshot([issuer.primary_ticker for issuer in universe]))
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.caption(
                f"Retrieved at {data['retrieved_at'].iloc[0]} from yfinance; values may be delayed."
            )
        except Exception as error:
            st.warning(f"Market data is temporarily unavailable: {type(error).__name__}")
