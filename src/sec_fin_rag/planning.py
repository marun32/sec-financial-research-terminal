"""Validate an LLM's research intent before it can constrain retrieval."""

from __future__ import annotations

import re
from typing import Any, Literal, cast

from .models import Issuer, ResearchPlan
from .query import infer_filing_form, infer_financial_metrics, resolve_tickers

_OPERATIONS = {"lookup", "trend", "yoy", "narrative", "mixed", "market", "unsupported"}
_BASES = {"annual", "quarterly", "ytd", "unspecified"}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _explicit_years(question: str, snapshot_year: int) -> list[int]:
    range_match = re.search(
        r"\b(20\d{2})\s*(?:to|through|until|[-–])\s*(20\d{2})\b", question, re.I
    )
    if range_match:
        first, last = (int(value) for value in range_match.groups())
        if 2010 <= first <= last <= snapshot_year and last - first < 11:
            return list(range(first, last + 1))
    return sorted(
        {
            int(value)
            for value in re.findall(r"(?<!\d)20\d{2}(?!\d)", question)
            if 2010 <= int(value) <= snapshot_year
        }
    )


def _relative_year_count(question: str) -> int:
    match = re.search(
        r"\b(?:last|latest|past|previous)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:fiscal\s+)?years?\b",
        question,
        re.I,
    )
    if not match:
        return 0
    token = match.group(1).lower()
    count = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    return min(count, 10)


def validate_research_plan(
    raw: dict[str, Any], question: str, issuers: list[Issuer]
) -> ResearchPlan:
    """Constrain model output to the issuer universe and explicit question semantics."""
    snapshot_year = int(issuers[0].snapshot_date[:4]) if issuers else 2026
    explicit_years = _explicit_years(question, snapshot_year)
    relative_count = _relative_year_count(question)
    model_years = raw.get("years", [])
    if not isinstance(model_years, list):
        model_years = []
    years = explicit_years or (
        []
        if relative_count
        else sorted(
            {
                int(year)
                for year in model_years
                if str(year).isdigit() and 2010 <= int(year) <= snapshot_year
            }
        )[:11]
    )
    if not years and not relative_count:
        candidate = raw.get("last_n_completed_years", 0)
        relative_count = min(max(int(candidate), 0), 10) if str(candidate).isdigit() else 0

    explicit_tickers = resolve_tickers(question, issuers)
    allowed = {issuer.primary_ticker for issuer in issuers}
    model_tickers = raw.get("tickers", [])
    if not isinstance(model_tickers, list):
        model_tickers = []
    tickers = explicit_tickers or [str(t) for t in model_tickers if str(t) in allowed]

    explicit_metrics = infer_financial_metrics(question)
    model_metrics = raw.get("metrics", [])
    if not isinstance(model_metrics, list):
        model_metrics = []
    metric_names = {
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "assets",
        "liabilities",
        "cash",
        "debt",
        "diluted_eps",
        "capital_expenditure",
    }
    metrics = explicit_metrics or [
        str(metric) for metric in model_metrics if metric in metric_names
    ]

    operation = str(raw.get("operation", "lookup"))
    if operation not in _OPERATIONS:
        operation = "lookup"
    lowered = question.lower()
    if re.search(r"\b(?:yoy|year[ -]over[ -]year)\b", lowered):
        operation = "yoy"
    elif (len(years) > 1 or relative_count > 1) and operation == "lookup":
        operation = "trend"

    basis = str(raw.get("period_basis", "unspecified"))
    if basis not in _BASES:
        basis = "unspecified"
    quarter_match = re.search(r"\bQ([1-4])\b", question, re.I)
    model_quarter = raw.get("quarter")
    quarter = (
        int(quarter_match.group(1))
        if quarter_match
        else (int(str(model_quarter)) if str(model_quarter) in {"1", "2", "3", "4"} else None)
    )
    explicit_form = infer_filing_form(question)
    if explicit_form == "10-K":
        basis = "annual"
    elif "ytd" in lowered or "year to date" in lowered or "year-to-date" in lowered:
        basis = "ytd"
    elif explicit_form == "10-Q" or quarter_match or "quarter" in lowered:
        basis = "quarterly"
    elif basis == "unspecified" and (len(years) > 1 or relative_count > 1):
        basis = "annual"
    elif basis == "unspecified" and operation == "yoy" and "quarter" not in lowered:
        basis = "annual"

    needs_filing_text = bool(raw.get("needs_filing_text", False)) or any(
        term in lowered for term in ("why", "explain", "driver", "risk", "management", "strategy")
    )
    return ResearchPlan(
        tickers=list(dict.fromkeys(tickers)),
        metrics=list(dict.fromkeys(metrics)),
        years=years,
        last_n_completed_years=relative_count,
        period_basis=cast(Literal["annual", "quarterly", "ytd", "unspecified"], basis),
        quarter=quarter,
        operation=cast(
            Literal["lookup", "trend", "yoy", "narrative", "mixed", "market", "unsupported"],
            operation,
        ),
        needs_filing_text=needs_filing_text,
    )
