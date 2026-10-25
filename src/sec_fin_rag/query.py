from __future__ import annotations

import re
from typing import Any, Literal

from .models import Issuer

FilingForm = Literal["10-K", "10-Q"]

FINANCIAL_METRIC_TERMS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales"),
    "net_income": ("net income", "net profit", "profit", "earnings"),
    "operating_income": ("operating income", "operating margin"),
    "operating_cash_flow": ("cash flow", "cash generation"),
    "assets": ("assets",),
    "liabilities": ("liabilities",),
    "cash": ("cash",),
    "debt": ("debt", "leverage"),
    "diluted_eps": ("eps", "earnings per share"),
}

_ANNUAL_PATTERNS = (
    r"\b10[ -]?k\b",
    r"\bannual (?:filings?|reports?|periods?|results?)\b",
    r"\blatest annual\b",
    r"\bfiscal year(?: end)?\b",
    r"\byear[ -]end\b",
    r"\bannual filings?\b",
)
_QUARTERLY_PATTERNS = (
    r"\b10[ -]?q\b",
    r"\bquarterly (?:filings?|reports?|periods?|results?)\b",
    r"\blatest quarter\b",
    r"\bfiscal quarter\b",
)


def _contains_phrase(question: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", question, flags=re.IGNORECASE))


def resolve_tickers(question: str, issuers: list[Issuer]) -> list[str]:
    """Resolve company names, aliases, and share classes to indexed primary tickers."""
    resolved: list[str] = []
    for issuer in issuers:
        ticker_variants = {
            variant for ticker in issuer.tickers for variant in (ticker, ticker.replace("-", "."))
        }
        names = {issuer.name, *issuer.aliases, *ticker_variants}
        if any(_contains_phrase(question, name) for name in names):
            resolved.append(issuer.primary_ticker)
    return resolved


def infer_filing_form(question: str) -> FilingForm | None:
    """Infer a filing form only when the user uses explicit annual or quarterly language."""
    if any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in _ANNUAL_PATTERNS):
        return "10-K"
    if any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in _QUARTERLY_PATTERNS):
        return "10-Q"
    return None


def infer_year(question: str) -> str | None:
    """Extract an explicit four-digit reporting year from a question."""
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", question)
    return match.group(1) if match else None


def infer_financial_metrics(question: str) -> list[str]:
    """Resolve supported logical metrics explicitly named in a question."""
    lowered = question.lower()
    return [
        metric
        for metric, terms in FINANCIAL_METRIC_TERMS.items()
        if any(term in lowered for term in terms)
    ]


def with_inferred_form(question: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill an unspecified form filter without overriding an explicit UI selection."""
    resolved = dict(filters or {})
    if resolved.get("form") in {None, "", "All"}:
        inferred = infer_filing_form(question)
        if inferred:
            resolved["form"] = inferred
    return resolved


def with_inferred_year(question: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill an unspecified reporting-year filter from natural-language input."""
    resolved = dict(filters or {})
    if resolved.get("year") in {None, "", "All"}:
        inferred = infer_year(question)
        if inferred:
            resolved["year"] = inferred
    return resolved


def pinecone_filter(tickers: list[str], filters: dict[str, Any]) -> dict[str, Any]:
    """Translate resolved query intent into Pinecone metadata-filter syntax."""
    query_filter: dict[str, Any] = {}
    if tickers:
        query_filter["ticker"] = {"$in": tickers}
    form = filters.get("form")
    if form and form != "All":
        query_filter["form"] = {"$in": [form, f"{form}/A"]}
    year = filters.get("year")
    if year and year != "All":
        query_filter["report_year"] = {"$eq": int(year)}
    return query_filter
