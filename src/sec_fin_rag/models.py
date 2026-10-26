from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict


@dataclass(frozen=True)
class Issuer:
    name: str
    tickers: tuple[str, ...]
    cik: int
    sector: str
    snapshot_date: str
    aliases: tuple[str, ...] = ()

    @property
    def primary_ticker(self) -> str:
        return self.tickers[0]


@dataclass(frozen=True)
class Filing:
    issuer: str
    ticker: str
    cik: int
    form: str
    filing_date: str
    report_date: str
    accession: str
    primary_document: str
    source_url: str
    snapshot_date: str = ""
    local_path: str = ""
    is_amendment: bool = False


@dataclass(frozen=True)
class DocumentBlock:
    block_id: str
    text: str
    block_type: Literal["text", "table"]
    section: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    strategy: Literal["fixed", "semantic"]
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    issuer: str
    form: str
    filing_date: str
    section: str
    source_url: str
    chunk_id: str
    source_number: int = 0
    source_type: Literal["filing", "xbrl", "market"] = "filing"


@dataclass(frozen=True)
class EvidenceItem:
    text: str
    citation: Citation


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class ResearchPlan(TypedDict):
    tickers: list[str]
    metrics: list[str]
    years: list[int]
    last_n_completed_years: int
    period_basis: Literal["annual", "quarterly", "ytd", "unspecified"]
    quarter: int | None
    operation: Literal["lookup", "trend", "yoy", "narrative", "mixed", "market", "unsupported"]
    needs_filing_text: bool


class GraphState(TypedDict, total=False):
    question: str
    resolved_question: str
    plan: ResearchPlan
    route: Literal["market", "fundamentals", "filings", "mixed", "unsupported"]
    tickers: list[str]
    filters: dict[str, Any]
    input_tickers: list[str]
    input_filters: dict[str, Any]
    conversation_context: dict[str, Any]
    retrieved: list[RetrievedChunk]
    structured_evidence: list[EvidenceItem]
    answer: str
    citations: list[Citation]
    citation_validation: dict[str, Any]
    error: str
