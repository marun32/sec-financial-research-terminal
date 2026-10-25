# ruff: noqa: E501
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from sec_fin_rag.ui import issuers, page_setup, runtime

page_setup(
    "Research Assistant",
    "Ask financial and filing questions. Answers are synthesized from SEC evidence with traceable references.",
    eyebrow="FIN AI · EVIDENCE WORKSPACE",
)

universe = issuers()
choices = {issuer.primary_ticker: issuer.name for issuer in universe}
query_company = str(st.query_params.get("company", ""))
default_companies = [query_company] if query_company in choices else []

with st.expander("Refine search (optional)", expanded=False):
    st.caption(
        "The assistant normally infers companies, filing type, and evidence route from your "
        "question. Use these controls only when you need to constrain the search."
    )
    selected = st.multiselect(
        "Limit to companies",
        choices,
        default=default_companies,
        format_func=lambda ticker: f"{ticker} · {choices[ticker]}",
        placeholder="Automatically inferred from your question",
    )
    filter_cols = st.columns(2)
    with filter_cols[0]:
        form = st.selectbox("Filing type", ["All", "10-K", "10-Q"])
    with filter_cols[1]:
        year = st.selectbox("Reporting year", ["All", "2026", "2025", "2024", "2023"])
    with st.popover("Advanced retrieval"):
        strategy = st.selectbox(
            "Retrieval strategy",
            ["auto", "ensemble", "semantic-v1", "fixed-v1"],
            help="Auto is recommended. Manual strategies are retained for retrieval analysis.",
        )


def display_answer(message: dict[str, Any]) -> None:
    st.markdown(str(message["content"]))
    references = message.get("references", [])
    if references:
        with st.expander(f"References ({len(references)})", expanded=False):
            show_excerpts = st.toggle(
                "Show evidence excerpts",
                value=False,
                key=f"excerpts-{message.get('message_id', id(message))}",
            )
            for reference in references:
                number = reference["number"]
                label = (
                    f"**[{number}] {reference['issuer']}** · {reference['form']} · "
                    f"{reference['filing_date']} · {reference['section']}"
                )
                if reference.get("source_url"):
                    st.markdown(f"{label} · [SEC source ↗]({reference['source_url']})")
                else:
                    st.markdown(label)
                if show_excerpts and reference.get("excerpt"):
                    st.caption(reference["excerpt"])
    if message.get("metadata"):
        metadata = message["metadata"]
        scope = ", ".join(metadata.get("tickers", [])) or "all indexed companies"
        st.markdown(
            f'<div class="answer-meta">{len(references)} references · '
            f"{metadata.get('route', 'research')} evidence · scope: {scope} · "
            f"generated {metadata.get('generated_at', '')}</div>",
            unsafe_allow_html=True,
        )


if "research_messages" not in st.session_state:
    st.session_state.research_messages = []
if "research_thread_id" not in st.session_state:
    st.session_state.research_thread_id = uuid.uuid4().hex

pending_prompt = ""
linked_prompt = str(st.query_params.get("prompt", ""))
if not st.session_state.research_messages:
    st.markdown("#### Start with an analyst question")
    suggestions = [
        "Compare revenue growth and operating margins for Apple and Microsoft.",
        "What material risks does NVIDIA disclose in its latest annual filing?",
        "Explain Amazon's cash-flow trend across recent annual periods.",
    ]
    if linked_prompt:
        suggestions.insert(0, linked_prompt)
    suggestion_cols = st.columns(min(3, len(suggestions)))
    for index, suggestion in enumerate(suggestions[:3]):
        with suggestion_cols[index]:
            if st.button(suggestion, key=f"suggestion-{index}", width="stretch"):
                pending_prompt = suggestion
    st.caption(
        "Ask about reported performance, risk factors, management discussion, trends, or cross-company comparisons."
    )

for message in st.session_state.research_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_answer(message)
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask a question grounded in SEC evidence…") or pending_prompt
if question:
    st.session_state.research_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Reviewing filings and checking evidence…"):
                result = runtime().graph.invoke(
                    question,
                    tickers=selected,
                    filters={"form": form, "year": year, "namespace": strategy},
                    thread_id=st.session_state.research_thread_id,
                )
            answer = re.sub(r"\[Source\s+(\d+)\]", r"[\1]", str(result["answer"]))
            excerpts = {item.chunk_id: item.text for item in result.get("retrieved", [])}
            excerpts.update(
                {
                    item.citation.chunk_id: item.text
                    for item in result.get("structured_evidence", [])
                }
            )
            references = [
                {
                    "number": citation.source_number,
                    "issuer": citation.issuer,
                    "form": citation.form,
                    "filing_date": citation.filing_date,
                    "section": citation.section,
                    "source_url": citation.source_url,
                    "excerpt": excerpts.get(citation.chunk_id, ""),
                }
                for citation in result.get("citations", [])
            ]
            assistant_message = {
                "message_id": datetime.now(UTC).isoformat(),
                "role": "assistant",
                "content": answer,
                "references": references,
                "metadata": {
                    "route": result.get("route", "research"),
                    "tickers": result.get("tickers", selected),
                    "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                },
            }
            display_answer(assistant_message)
        except Exception as error:
            assistant_message = {
                "role": "assistant",
                "content": (
                    "The research service is temporarily unavailable. Your SEC artifacts remain "
                    f"intact; verify service connectivity and try again. ({type(error).__name__})"
                ),
                "references": [],
                "metadata": {},
            }
            st.error(assistant_message["content"])
    st.session_state.research_messages.append(assistant_message)

if st.session_state.research_messages and st.button("Clear conversation", type="tertiary"):
    st.session_state.research_messages = []
    st.session_state.research_thread_id = uuid.uuid4().hex
    st.rerun()
