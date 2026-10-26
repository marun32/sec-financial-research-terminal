from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

PERIOD_LABELS = {
    "instant": "At period end",
    "three_months": "3 months",
    "six_months": "6 months",
    "nine_months": "9 months",
    "annual": "12 months",
    "other": "Other duration",
}


def fact_period(start: Any, end: Any) -> tuple[int | None, str]:
    """Classify a fact's own duration, independently of the filing's FY/FP labels."""
    if start is None or pd.isna(start):
        return None, "instant"
    try:
        first = date.fromisoformat(str(start)[:10])
        last = date.fromisoformat(str(end)[:10])
    except ValueError:
        return None, "other"
    days = (last - first).days + 1
    if days <= 0:
        return days, "other"
    if 70 <= days <= 110:
        return days, "three_months"
    if 150 <= days <= 215:
        return days, "six_months"
    if 235 <= days <= 315:
        return days, "nine_months"
    if 330 <= days <= 400:
        return days, "annual"
    return days, "other"


def with_fact_periods(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fact-level duration metadata to normalized Company Facts records."""
    if {"duration_days", "period_type"}.issubset(frame.columns):
        return frame
    result = frame.copy()
    starts = result["start"] if "start" in result else [None] * len(result)
    periods = [fact_period(start, end) for start, end in zip(starts, result["end"], strict=True)]
    result["duration_days"] = [days for days, _ in periods]
    result["period_type"] = [kind for _, kind in periods]
    return result


def annual_fact_rows(frame: pd.DataFrame) -> pd.DataFrame:
    facts = with_fact_periods(frame)
    return facts[facts["period_type"].isin(["annual", "instant"])].copy()


def quarter_fact_rows(frame: pd.DataFrame) -> pd.DataFrame:
    facts = with_fact_periods(frame)
    return facts[facts["period_type"].isin(["three_months", "instant"])].copy()
