from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .citations import number_citations, validate_answer_citations
from .models import Citation, EvidenceItem, GraphState, Issuer, RetrievedChunk
from .query import (
    infer_financial_metrics,
    resolve_tickers,
    with_inferred_form,
    with_inferred_year,
)

MARKET_TERMS = {"price", "trading", "market cap", "today", "daily", "volume"}
FUNDAMENTAL_TERMS = {
    "revenue",
    "income",
    "margin",
    "assets",
    "liabilities",
    "cash flow",
    "eps",
    "ratio",
    "cagr",
    "financial information",
    "financial overview",
    "financial performance",
    "financial results",
    "financials",
    "performance",
}
FILING_TERMS = {
    "risk",
    "strategy",
    "management",
    "disclosure",
    "competition",
    "explain",
    "why",
    "10-k",
    "10-q",
    "annual filing",
    "annual report",
    "annual filings",
    "quarterly filing",
    "quarterly report",
    "fiscal year",
    "year end",
    "year-end",
}
ADVICE_TERMS = {
    "should i buy",
    "should i sell",
    "invest my",
    "retirement savings",
    "price prediction",
    "exact stock price",
    "confidential",
    "not yet announced",
}


def _has_route_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered for term in MARKET_TERMS | FUNDAMENTAL_TERMS | FILING_TERMS | ADVICE_TERMS
    )


def classify_route(
    question: str,
) -> Literal["market", "fundamentals", "filings", "mixed", "unsupported"]:
    lowered = question.lower()
    if any(term in lowered for term in ADVICE_TERMS):
        return "unsupported"
    market = any(term in lowered for term in MARKET_TERMS)
    fundamental = any(term in lowered for term in FUNDAMENTAL_TERMS)
    filing = any(term in lowered for term in FILING_TERMS)
    if market and (fundamental or filing):
        return "mixed"
    if market:
        return "market"
    if fundamental and filing:
        return "mixed"
    if fundamental:
        return "fundamentals"
    if filing:
        return "filings"
    return "filings"


def extract_tickers(question: str, known_tickers: set[str]) -> list[str]:
    words = set(re.findall(r"\b[A-Za-z][A-Za-z.-]{0,7}\b", question.upper()))
    return sorted(words & {ticker.upper() for ticker in known_tickers})


class ResearchGraph:
    def __init__(
        self,
        issuers: list[Issuer],
        retrieve: Callable[[str, list[str], dict[str, Any]], list[RetrievedChunk]],
        fundamentals: Callable[[str, list[str], dict[str, Any]], list[EvidenceItem] | str],
        market: Callable[[list[str]], list[EvidenceItem] | str],
        generate: Callable[[str, list[RetrievedChunk], list[EvidenceItem]], str],
    ):
        self.issuers = issuers
        self.retrieve_fn = retrieve
        self.fundamentals_fn = fundamentals
        self.market_fn = market
        self.generate_fn = generate
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules={
                ("sec_fin_rag.models", "Citation"),
                ("sec_fin_rag.models", "EvidenceItem"),
                ("sec_fin_rag.models", "RetrievedChunk"),
            }
        )
        self.checkpointer = InMemorySaver(serde=serializer)
        self.compiled = self._build()

    def _build(self) -> Any:
        graph = StateGraph(GraphState)
        graph.add_node("route", self._route)
        graph.add_node("market", self._market)
        graph.add_node("fundamentals", self._fundamentals)
        graph.add_node("filings", self._filings)
        graph.add_node("mixed", self._mixed)
        graph.add_node("unsupported", self._unsupported)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {
                "market": "market",
                "fundamentals": "fundamentals",
                "filings": "filings",
                "mixed": "mixed",
                "unsupported": "unsupported",
            },
        )
        for node in ("market", "fundamentals", "filings", "mixed"):
            graph.add_edge(node, "answer")
        graph.add_edge("unsupported", END)
        graph.add_edge("answer", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _route(self, state: GraphState) -> GraphState:
        question = state["question"]
        previous = state.get("conversation_context", {})
        explicit_tickers = resolve_tickers(question, self.issuers)
        manual_tickers = state.get("input_tickers", [])
        tickers = manual_tickers or explicit_tickers or list(previous.get("tickers", []))

        input_filters = dict(state.get("input_filters", {}))
        filters = with_inferred_year(question, with_inferred_form(question, input_filters))
        if filters.get("year") in {None, "", "All"} and previous.get("year"):
            filters["year"] = previous["year"]
        if filters.get("form") in {None, "", "All"} and previous.get("form") not in {
            None,
            "",
            "All",
        }:
            filters["form"] = previous["form"]

        route = classify_route(question)
        if not _has_route_signal(question) and previous.get("route"):
            route = previous["route"]

        explicit_metrics = infer_financial_metrics(question)
        metrics = explicit_metrics or list(previous.get("metrics", []))
        context = {
            "tickers": tickers,
            "route": route,
            "year": filters.get("year") if filters.get("year") != "All" else None,
            "form": filters.get("form") if filters.get("form") != "All" else None,
            "metrics": metrics,
        }
        context_parts = []
        if tickers:
            context_parts.append(f"companies: {', '.join(tickers)}")
        if context["year"]:
            context_parts.append(f"reporting year: {context['year']}")
        if metrics:
            context_parts.append(f"metrics: {', '.join(metrics)}")
        context_parts.append(f"evidence route: {route}")
        resolved_question = question
        if context_parts:
            resolved_question += "\nResolved research context — " + "; ".join(context_parts) + "."

        return {
            "route": route,
            "tickers": tickers,
            "filters": filters,
            "resolved_question": resolved_question,
            "conversation_context": context,
            "retrieved": [],
            "structured_evidence": [],
            "answer": "",
            "citations": [],
            "citation_validation": {},
        }

    def _market(self, state: GraphState) -> GraphState:
        return {
            "structured_evidence": self._normalize_evidence(
                self.market_fn(state.get("tickers", [])), "market"
            )
        }

    def _fundamentals(self, state: GraphState) -> GraphState:
        resolved_question = state.get("resolved_question", state["question"])
        return {
            "structured_evidence": self._normalize_evidence(
                self.fundamentals_fn(
                    resolved_question,
                    state.get("tickers", []),
                    state.get("filters", {}),
                ),
                "xbrl",
            ),
            # Structured facts anchor the numbers; filing RAG supplies management explanation and
            # disclosure context for the same financial question.
            "retrieved": self.retrieve_fn(
                resolved_question,
                state.get("tickers", []),
                state.get("filters", {}),
            ),
        }

    def _filings(self, state: GraphState) -> GraphState:
        return {
            "retrieved": self.retrieve_fn(
                state.get("resolved_question", state["question"]),
                state.get("tickers", []),
                state.get("filters", {}),
            )
        }

    def _mixed(self, state: GraphState) -> GraphState:
        resolved_question = state.get("resolved_question", state["question"])
        structured = self._normalize_evidence(
            self.fundamentals_fn(
                resolved_question,
                state.get("tickers", []),
                state.get("filters", {}),
            ),
            "xbrl",
        )
        market_evidence: list[EvidenceItem] = []
        if any(term in state["question"].lower() for term in MARKET_TERMS):
            try:
                market_evidence = self._normalize_evidence(
                    self.market_fn(state.get("tickers", [])), "market"
                )
            except Exception:
                market_evidence = []
        return {
            "structured_evidence": structured + market_evidence,
            "retrieved": self.retrieve_fn(
                resolved_question, state.get("tickers", []), state.get("filters", {})
            ),
        }

    def _unsupported(self, state: GraphState) -> GraphState:
        return {
            "answer": (
                "I can compare documented financial evidence and transparent scorecards, but I "
                "cannot provide personalized buy, sell, or portfolio recommendations."
            ),
            "citations": [],
        }

    def _answer(self, state: GraphState) -> GraphState:
        retrieved = state.get("retrieved", [])
        structured = state.get("structured_evidence", [])
        if not retrieved and not structured:
            return {
                "answer": "I could not find sufficient evidence in the configured sources.",
                "citations": [],
            }
        answer = self.generate_fn(
            state.get("resolved_question", state["question"]), retrieved, structured
        )
        citations = [
            Citation(
                issuer=str(item.metadata.get("issuer", "")),
                form=str(item.metadata.get("form", "")),
                filing_date=str(item.metadata.get("filing_date", "")),
                section=str(item.metadata.get("section", "")),
                source_url=str(item.metadata.get("source_url", "")),
                chunk_id=item.chunk_id,
            )
            for item in retrieved
        ] + [item.citation for item in structured]
        numbered = number_citations(citations)
        answer, validation = validate_answer_citations(answer, numbered)
        if not bool(validation["is_valid"]):
            fail_closed = (
                "I could not produce an answer with verifiable inline citations from the "
                "available evidence."
            )
            validation["failed_closed"] = True
            return {
                "answer": fail_closed,
                "citations": [],
                "citation_validation": validation,
            }
        return {
            "answer": answer,
            "citations": numbered,
            "citation_validation": validation,
        }

    @staticmethod
    def _normalize_evidence(
        value: list[EvidenceItem] | str, source_type: Literal["xbrl", "market"]
    ) -> list[EvidenceItem]:
        if not isinstance(value, str):
            return value
        if not value:
            return []
        return [
            EvidenceItem(
                text=value,
                citation=Citation(
                    issuer="Structured data",
                    form="SEC XBRL" if source_type == "xbrl" else "Delayed market data",
                    filing_date="",
                    section="Structured evidence",
                    source_url="",
                    chunk_id=f"{source_type}:legacy",
                    source_type=source_type,
                ),
            )
        ]

    def invoke(
        self,
        question: str,
        tickers: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        *,
        thread_id: str | None = None,
    ) -> GraphState:
        conversation_id = thread_id or uuid.uuid4().hex
        return cast(
            GraphState,
            self.compiled.invoke(
                {
                    "question": question,
                    "input_tickers": tickers or [],
                    "input_filters": filters or {},
                },
                config={"configurable": {"thread_id": conversation_id}},
            ),
        )
