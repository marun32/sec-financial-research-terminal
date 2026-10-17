from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import Settings
from .graph import ResearchGraph
from .manifest import load_issuers
from .market import market_snapshot
from .models import RetrievedChunk
from .providers import LocalReranker, NebiusClient
from .retrieval import PineconeStore, rerank_chunks
from .sparse import SparseBM25Encoder


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.issuers = load_issuers(settings.manifest_path)
        self.nebius = NebiusClient(settings)
        self.sparse = SparseBM25Encoder.load(settings.app_data_dir / "bm25.json")
        self.store = PineconeStore(settings, self.nebius, self.sparse)
        self.reranker = LocalReranker()
        path = settings.app_data_dir / "fundamentals.parquet"
        self.fundamentals = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        known = {ticker for issuer in self.issuers for ticker in issuer.tickers}
        self.graph = ResearchGraph(
            known_tickers=known,
            retrieve=self.retrieve,
            fundamentals=self.fundamental_context,
            market=self.market_context,
            generate=self.generate,
        )

    def retrieve(
        self, question: str, tickers: list[str], filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        query_filters: dict[str, Any] = {}
        if tickers:
            query_filters["ticker"] = {"$in": tickers}
        if filters.get("form") and filters["form"] != "All":
            query_filters["form"] = {"$in": [filters["form"], f"{filters['form']}/A"]}
        if filters.get("year") and filters["year"] != "All":
            query_filters["report_year"] = {"$eq": int(filters["year"])}
        namespace = filters.get("namespace", "semantic-v1")
        candidates = self.store.query(question, namespace, query_filters, top_k=30)
        return rerank_chunks(question, candidates, self.reranker, candidate_count=20, top_n=8)

    def fundamental_context(self, question: str, tickers: list[str]) -> str:
        if self.fundamentals.empty:
            return "Structured SEC XBRL data is unavailable."
        frame = self.fundamentals
        if tickers:
            frame = frame[frame["ticker"].isin(tickers)]
        recent = frame.sort_values("filed").groupby(["ticker", "metric"], as_index=False).tail(12)
        records = recent[
            ["ticker", "metric", "value", "unit", "end", "filed", "form", "accession"]
        ].to_dict("records")
        return "Recent structured SEC XBRL facts: " + json.dumps(records[-200:], default=str)

    def market_context(self, tickers: list[str]) -> str:
        if not tickers:
            return "Select or mention a ticker to retrieve delayed market context."
        return "Delayed market context: " + json.dumps(market_snapshot(tickers), default=str)

    def generate(
        self, question: str, retrieved: list[RetrievedChunk], structured_context: str
    ) -> str:
        contexts = [{**item.metadata, "text": item.text} for item in retrieved]
        if structured_context:
            contexts.append(
                {
                    "issuer": "Structured data",
                    "form": "SEC XBRL / market context",
                    "filing_date": "See evidence",
                    "section": "Structured facts",
                    "text": structured_context,
                }
            )
        return self.nebius.answer(question, contexts)
