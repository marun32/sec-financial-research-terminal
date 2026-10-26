from __future__ import annotations

import json
from typing import Any, cast

import pandas as pd
from langsmith import traceable

from .config import Settings
from .fact_periods import PERIOD_LABELS, annual_fact_rows, quarter_fact_rows, with_fact_periods
from .filing_catalog import sec_filing_index_url
from .graph import ResearchGraph
from .manifest import load_issuers
from .market import market_snapshot
from .models import Citation, EvidenceItem, ResearchPlan, RetrievedChunk
from .planning import validate_research_plan
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
            planner=self.plan_question,
        )

    def plan_question(self, question: str, context: dict[str, Any]) -> ResearchPlan:
        issuer_names = [
            f"{issuer.primary_ticker}: {issuer.name} ({', '.join(issuer.aliases)})"
            for issuer in self.issuers
        ]
        raw = self.nebius.plan_question(question, context, issuer_names)
        return validate_research_plan(raw, question, self.issuers)

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
        frame = with_fact_periods(self.fundamentals).copy()
        if "source_report_end" not in frame:
            frame["source_report_end"] = None
        if "source_period_match" not in frame:
            frame["source_period_match"] = False
        if tickers:
            frame = frame[frame["ticker"].isin(tickers)]
        requested_metrics = infer_financial_metrics(question)
        if requested_metrics:
            frame = frame[frame["metric"].isin(requested_metrics)]
        planned_years = [int(year) for year in filters.get("_planned_years", [])]
        last_n = int(filters.get("_last_n_completed_years", 0))
        basis = str(filters.get("_period_basis", "unspecified"))
        multi_period_plan = (len(planned_years) > 1 or last_n > 1) and basis in {
            "annual",
            "quarterly",
            "ytd",
        }
        coverage_note = ""
        if multi_period_plan:
            if basis == "annual":
                selected = annual_fact_rows(frame[frame["form"].isin(["10-K", "10-K/A"])])
            else:
                selected = frame[frame["form"].isin(["10-Q", "10-Q/A"])]
                if basis == "quarterly":
                    selected = quarter_fact_rows(selected)
                else:
                    selected = selected[
                        selected["period_type"].isin(["three_months", "six_months", "nine_months"])
                    ]
                quarter = filters.get("_quarter")
                if quarter and "fiscal_period" in selected:
                    selected = selected[selected["fiscal_period"] == f"Q{quarter}"]
                    if basis == "ytd":
                        durations = {1: "three_months", 2: "six_months", 3: "nine_months"}
                        selected = selected[selected["period_type"] == durations.get(int(quarter))]
            selected = selected.sort_values(["end", "filed"]).drop_duplicates(
                ["ticker", "metric", "start", "end"], keep="last"
            )
            if planned_years:
                selected = selected[selected["end"].str[:4].astype(int).isin(planned_years)]
            elif last_n:
                if basis == "annual":
                    selected = selected.groupby(["ticker", "metric"], group_keys=False).tail(last_n)
                else:
                    latest_years = sorted(selected["end"].str[:4].unique(), reverse=True)[:last_n]
                    selected = selected[selected["end"].str[:4].isin(latest_years)]
            recent = selected.copy()
            if planned_years:
                found = sorted(selected["end"].str[:4].astype(int).unique().tolist())
                missing = sorted(set(planned_years) - set(found))
                coverage_note = (
                    f"Requested {basis} period-end years: {planned_years}. "
                    f"{basis.capitalize()} facts supplied: {found}. "
                    f"{basis.capitalize()} facts not supplied: {missing}. "
                    "Do not claim that a filing lacks a figure merely because it was not "
                    "supplied. Keep interim and full-year comparisons separate. If matching "
                    "current/prior interim facts follow, calculate their same-duration YoY "
                    "change separately. "
                )
                if basis == "annual" and missing and filters.get("_operation") == "yoy":
                    recent = pd.concat(
                        [recent, self._latest_comparable_interim(frame, missing)],
                        ignore_index=True,
                    )
        else:
            year = filters.get("year")
            if year and year != "All":
                frame = frame[frame["end"].astype(str).str.startswith(str(year))]
            form = filters.get("form")
            if form and form != "All":
                frame = frame[frame["form"].isin([form, f"{form}/A"])]
                if form == "10-K":
                    frame = annual_fact_rows(frame)
            recent = (
                frame.sort_values(["end", "duration_days", "filed"])
                .groupby(["ticker", "metric"], as_index=False)
                .tail(8)
            )
        issuer_by_ticker = {issuer.primary_ticker: issuer for issuer in self.issuers}
        evidence: list[EvidenceItem] = []
        for (ticker, accession), rows in recent.groupby(["ticker", "accession"]):
            issuer = issuer_by_ticker[str(ticker)]
            filing_url = sec_filing_index_url(issuer.cik, str(accession))
            records = rows.sort_values(["metric", "end", "duration_days", "filed"])[
                [
                    "ticker",
                    "metric",
                    "value",
                    "unit",
                    "start",
                    "end",
                    "duration_days",
                    "period_type",
                    "filed",
                    "form",
                    "accession",
                    "source_report_end",
                    "source_period_match",
                ]
            ].to_dict("records")
            for record in records:
                for field in ("start", "duration_days", "source_report_end"):
                    if pd.isna(record[field]):
                        record[field] = None
                record["fact_duration"] = PERIOD_LABELS.get(str(record["period_type"]), "Other")
            evidence.append(
                EvidenceItem(
                    text=coverage_note
                    + "Normalized SEC XBRL facts. Start/end and fact_duration describe the "
                    "individual fact; source_report_end identifies the source filing's report "
                    "period. A mismatch means this is a comparative fact. "
                    + json.dumps(records[-60:], default=str, allow_nan=False),
                    citation=Citation(
                        issuer=issuer.name,
                        form=str(rows.iloc[0]["form"]) if filing_url else "SEC XBRL Company Facts",
                        filing_date=str(rows["filed"].max()),
                        section="Structured fundamentals",
                        source_url=filing_url
                        or f"https://data.sec.gov/api/xbrl/companyfacts/CIK{issuer.cik:010d}.json",
                        chunk_id=f"xbrl:{issuer.cik}:{accession}",
                        source_type="xbrl",
                    ),
                )
            )
        return evidence

    @staticmethod
    def _latest_comparable_interim(frame: pd.DataFrame, missing_years: list[int]) -> pd.DataFrame:
        """Add current/prior YTD facts only when a requested annual year is incomplete."""
        quarterly = with_fact_periods(frame[frame["form"].isin(["10-Q", "10-Q/A"])])
        quarterly = quarterly[
            quarterly["period_type"].isin(["three_months", "six_months", "nine_months"])
        ]
        selected: list[pd.DataFrame] = []
        for (_ticker, _metric), group in quarterly.groupby(["ticker", "metric"]):
            for year in missing_years:
                current = group[group["end"].str[:4] == str(year)].sort_values(
                    ["end", "duration_days", "filed"]
                )
                if current.empty:
                    continue
                current_row = current.iloc[[-1]]
                prior = group[
                    (group["end"].str[:4] == str(year - 1))
                    & (group["period_type"] == current_row.iloc[0]["period_type"])
                    & (group["fiscal_period"] == current_row.iloc[0]["fiscal_period"])
                ].sort_values(["end", "filed"])
                if not prior.empty:
                    selected.extend([prior.iloc[[-1]], current_row])
        return pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0].copy()

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
