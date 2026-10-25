from __future__ import annotations

import json
from typing import Any, cast

import pandas as pd
from langsmith import traceable

from .config import Settings
from .graph import ResearchGraph
from .manifest import load_issuers
from .market import market_snapshot
from .models import Citation, EvidenceItem, RetrievedChunk
from .providers import LocalReranker, NebiusClient
from .query import infer_financial_metrics, pinecone_filter
from .retrieval import (
    PineconeStore,
    rerank_chunks,
    retrieve_candidates,
    select_retrieval_strategy,
)
from .sparse import load_sparse_encoders


def _safe_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "self"}


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.issuers = load_issuers(settings.manifest_path)
        self.nebius = NebiusClient(settings)
        self.sparse = load_sparse_encoders(settings.app_data_dir)
        self.store = PineconeStore(settings, self.nebius, self.sparse)
        self.reranker = LocalReranker()
        path = settings.app_data_dir / "fundamentals.parquet"
        self.fundamentals = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        self.graph = ResearchGraph(
            issuers=self.issuers,
            retrieve=self.retrieve,
            fundamentals=self.fundamental_evidence,
            market=self.market_evidence,
            generate=self.generate,
        )

    @traceable(  # type: ignore[untyped-decorator]
        name="hybrid_retrieve_and_rerank",
        run_type="retriever",
        process_inputs=_safe_trace_inputs,
    )
    def retrieve(
        self, question: str, tickers: list[str], filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        query_filters = pinecone_filter(tickers, filters)
        requested = str(filters.get("namespace", "auto"))
        decision = select_retrieval_strategy(question, tickers)
        strategy = decision.strategy if requested == "auto" else requested
        candidates = retrieve_candidates(
            self.store,
            question,
            strategy,
            query_filters,
            top_k=30,
            alpha=self.settings.retrieval_alpha,
        )
        return rerank_chunks(
            question,
            candidates,
            self.reranker,
            candidate_count=self.settings.rerank_candidate_count,
            top_n=self.settings.context_chunk_count,
            preserve_tickers=tickers,
        )

    def fundamental_evidence(
        self, question: str, tickers: list[str], filters: dict[str, Any]
    ) -> list[EvidenceItem]:
        if self.fundamentals.empty:
            return []
        frame = self.fundamentals
        if tickers:
            frame = frame[frame["ticker"].isin(tickers)]
        year = filters.get("year")
        if year and year != "All":
            frame = frame[frame["end"].astype(str).str.startswith(str(year))]
        form = filters.get("form")
        if form and form != "All":
            frame = frame[frame["form"].isin([form, f"{form}/A"])]
        requested_metrics = infer_financial_metrics(question)
        if requested_metrics:
            frame = frame[frame["metric"].isin(requested_metrics)]
        recent = (
            frame.sort_values(["end", "filed"])
            .groupby(["ticker", "metric"], as_index=False)
            .tail(6)
        )
        issuer_by_ticker = {issuer.primary_ticker: issuer for issuer in self.issuers}
        evidence: list[EvidenceItem] = []
        for ticker, rows in recent.groupby("ticker"):
            records = rows.sort_values(["metric", "end", "filed"])[
                ["ticker", "metric", "value", "unit", "end", "filed", "form", "accession"]
            ].to_dict("records")
            issuer = issuer_by_ticker[str(ticker)]
            evidence.append(
                EvidenceItem(
                    text="Relevant normalized SEC XBRL facts: "
                    + json.dumps(records[-60:], default=str),
                    citation=Citation(
                        issuer=issuer.name,
                        form="SEC XBRL Company Facts",
                        filing_date=str(rows["filed"].max()),
                        section="Structured fundamentals",
                        source_url=(
                            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{issuer.cik:010d}.json"
                        ),
                        chunk_id=f"xbrl:{issuer.cik}",
                        source_type="xbrl",
                    ),
                )
            )
        return evidence

    def market_evidence(self, tickers: list[str]) -> list[EvidenceItem]:
        if not tickers:
            return []
        issuer_by_ticker = {issuer.primary_ticker: issuer for issuer in self.issuers}
        return [
            EvidenceItem(
                text="Delayed market context: " + json.dumps(row, default=str),
                citation=Citation(
                    issuer=issuer_by_ticker[str(row["ticker"])].name,
                    form="Delayed yfinance context",
                    filing_date=str(row["retrieved_at"]),
                    section="Market snapshot",
                    source_url=f"https://finance.yahoo.com/quote/{row['ticker']}",
                    chunk_id=f"market:{row['ticker']}:{row['retrieved_at']}",
                    source_type="market",
                ),
            )
            for row in market_snapshot(tickers)
        ]

    @traceable(  # type: ignore[untyped-decorator]
        name="grounded_generation", run_type="chain", process_inputs=_safe_trace_inputs
    )
    def generate(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        structured_evidence: list[EvidenceItem],
    ) -> str:
        contexts = [{**item.metadata, "text": item.text} for item in retrieved]
        contexts.extend(
            {
                "issuer": item.citation.issuer,
                "form": item.citation.form,
                "filing_date": item.citation.filing_date,
                "section": item.citation.section,
                "text": item.text,
            }
            for item in structured_evidence
        )
        return cast(str, self.nebius.answer(question, contexts))
