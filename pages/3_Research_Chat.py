import streamlit as st

from sec_fin_rag.ui import issuers, page_setup, runtime

page_setup("Research Chat")
universe = issuers()
choices = {issuer.primary_ticker: issuer.name for issuer in universe}
selected = st.multiselect(
    "Limit to companies", choices, format_func=lambda ticker: f"{ticker} — {choices[ticker]}"
)
col1, col2, col3 = st.columns(3)
with col1:
    form = st.selectbox("Filing", ["All", "10-K", "10-Q"])
with col2:
    year = st.selectbox("Reporting year", ["All", "2026", "2025", "2024", "2023"])
with col3:
    strategy = st.selectbox("Chunk strategy", ["semantic-v1", "fixed-v1"])

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about performance, risks, trends, or disclosures…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding and checking evidence…"):
                result = runtime().graph.invoke(
                    question,
                    tickers=selected,
                    filters={"form": form, "year": year, "namespace": strategy},
                )
            st.markdown(result["answer"])
            if result.get("citations"):
                with st.expander("Sources"):
                    for citation in result["citations"]:
                        st.markdown(
                            f"- [{citation.issuer} — {citation.form}, {citation.filing_date}, "
                            f"{citation.section}]({citation.source_url})"
                        )
            answer = result["answer"]
        except Exception as error:
            answer = (
                f"The research service is unavailable ({type(error).__name__}). "
                "Run preflight and verify artifacts."
            )
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
