from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from .models import Citation, GraphState, RetrievedChunk

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
}
FILING_TERMS = {"risk", "strategy", "management", "disclosure", "competition", "explain", "why"}
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
        known_tickers: set[str],
        retrieve: Callable[[str, list[str], dict[str, Any]], list[RetrievedChunk]],
        fundamentals: Callable[[str, list[str]], str],
        market: Callable[[list[str]], str],
        generate: Callable[[str, list[RetrievedChunk], str], str],
    ):
        self.known_tickers = known_tickers
        self.retrieve_fn = retrieve
        self.fundamentals_fn = fundamentals
        self.market_fn = market
        self.generate_fn = generate
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
        return graph.compile()

    def _route(self, state: GraphState) -> GraphState:
        return {
            "route": classify_route(state["question"]),
            "tickers": state.get("tickers")
            or extract_tickers(state["question"], self.known_tickers),
        }

    def _market(self, state: GraphState) -> GraphState:
        return {"structured_context": self.market_fn(state.get("tickers", []))}

    def _fundamentals(self, state: GraphState) -> GraphState:
        return {
            "structured_context": self.fundamentals_fn(state["question"], state.get("tickers", []))
        }

    def _filings(self, state: GraphState) -> GraphState:
        return {
            "retrieved": self.retrieve_fn(
                state["question"], state.get("tickers", []), state.get("filters", {})
            )
        }

    def _mixed(self, state: GraphState) -> GraphState:
        structured = self.fundamentals_fn(state["question"], state.get("tickers", []))
        market_context = ""
        if any(term in state["question"].lower() for term in MARKET_TERMS):
            try:
                market_context = self.market_fn(state.get("tickers", []))
            except Exception:
                market_context = "Market context unavailable."
        return {
            "structured_context": f"{structured}\n{market_context}",
            "retrieved": self.retrieve_fn(
                state["question"], state.get("tickers", []), state.get("filters", {})
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
        if not retrieved and not state.get("structured_context"):
            return {
                "answer": "I could not find sufficient evidence in the configured sources.",
                "citations": [],
            }
        answer = self.generate_fn(state["question"], retrieved, state.get("structured_context", ""))
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
        ]
        return {"answer": answer, "citations": citations}

    def invoke(
        self, question: str, tickers: list[str] | None = None, filters: dict[str, Any] | None = None
    ) -> GraphState:
        return cast(
            GraphState,
            self.compiled.invoke(
                {"question": question, "tickers": tickers or [], "filters": filters or {}}
            ),
        )
