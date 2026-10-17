from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial

import numpy as np
import pandas as pd

SCORECARD_WEIGHTS: dict[str, dict[str, float]] = {
    "Growth": {"revenue_cagr_3y": 0.40, "revenue_growth_yoy": 0.30, "eps_growth": 0.30},
    "Profitability": {
        "operating_margin": 0.35,
        "net_margin": 0.25,
        "return_on_assets": 0.20,
        "fcf_margin": 0.20,
    },
    "Financial Strength": {
        "cash_to_debt": 0.30,
        "inverse_debt_to_assets": 0.25,
        "current_ratio": 0.25,
        "interest_coverage": 0.20,
    },
}


def _safe_div(numerator: float, denominator: float) -> float:
    return np.nan if not denominator or pd.isna(denominator) else numerator / denominator


def _growth(current: float, previous: float) -> float:
    return np.nan if pd.isna(previous) or previous == 0 else current / previous - 1


def _annual_value(
    company: pd.DataFrame, periods: list[float], metric: str, offset: int = 0
) -> float:
    if offset >= len(periods):
        return np.nan
    series = company[(company["metric"] == metric) & (company["fiscal_year"] == periods[offset])]
    return float(series.iloc[-1]["value"]) if not series.empty else np.nan


def derive_company_metrics(fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        return pd.DataFrame()
    annual = fundamentals[
        fundamentals["form"].isin(["10-K", "10-K/A"])
        & fundamentals["fiscal_period"].isin(["FY", None])
    ].copy()
    rows: list[dict[str, float | str]] = []
    for ticker, company in annual.groupby("ticker"):
        periods = sorted(company["fiscal_year"].dropna().unique(), reverse=True)
        if not periods:
            continue

        value = partial(_annual_value, company, list(periods))

        revenue, revenue_1y, revenue_3y = value("revenue"), value("revenue", 1), value("revenue", 3)
        eps, eps_1y = value("diluted_eps"), value("diluted_eps", 1)
        assets = value("assets")
        debt = value("debt")
        operating_income = value("operating_income")
        operating_cash = value("operating_cash_flow")
        capex = value("capital_expenditure")
        interest = value("interest_expense")
        rows.append(
            {
                "ticker": ticker,
                "issuer": str(company.iloc[-1]["issuer"]),
                "sector": str(company.iloc[-1]["sector"]),
                "fiscal_year": int(periods[0]),
                "revenue_cagr_3y": (
                    (revenue / revenue_3y) ** (1 / 3) - 1
                    if revenue > 0 and revenue_3y > 0
                    else np.nan
                ),
                "revenue_growth_yoy": _growth(revenue, revenue_1y),
                "eps_growth": _growth(eps, eps_1y),
                "operating_margin": _safe_div(operating_income, revenue),
                "net_margin": _safe_div(value("net_income"), revenue),
                "return_on_assets": _safe_div(value("net_income"), assets),
                "fcf_margin": _safe_div(operating_cash - capex, revenue),
                "cash_to_debt": _safe_div(value("cash"), debt),
                "inverse_debt_to_assets": -_safe_div(debt, assets),
                "current_ratio": _safe_div(value("current_assets"), value("current_liabilities")),
                "interest_coverage": _safe_div(operating_income, interest),
            }
        )
    return pd.DataFrame.from_records(rows)


def _winsorized_percentile(series: pd.Series) -> pd.Series:
    usable = series.replace([np.inf, -np.inf], np.nan)
    if usable.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index)
    low, high = usable.quantile([0.05, 0.95])
    return usable.clip(low, high).rank(pct=True) * 100


def score_companies(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    result = metrics.copy()
    for metric in {name for weights in SCORECARD_WEIGHTS.values() for name in weights}:
        result[f"{metric}_percentile"] = _winsorized_percentile(result[metric])
    for scorecard, weights in SCORECARD_WEIGHTS.items():
        scores = []
        coverages = []
        for _, row in result.iterrows():
            available: Mapping[str, float] = {
                metric: weight
                for metric, weight in weights.items()
                if not pd.isna(row[f"{metric}_percentile"])
            }
            total = sum(available.values())
            scores.append(
                sum(
                    row[f"{metric}_percentile"] * weight / total
                    for metric, weight in available.items()
                )
                if total
                else np.nan
            )
            coverages.append(total)
        result[scorecard] = scores
        result[f"{scorecard}_coverage"] = coverages
    result["Balanced"] = result[list(SCORECARD_WEIGHTS)].mean(axis=1, skipna=True)
    result["Balanced_coverage"] = result[[f"{name}_coverage" for name in SCORECARD_WEIGHTS]].mean(
        axis=1
    )
    for column in result.select_dtypes(include="number"):
        result[column] = result[column].map(
            lambda value: value if not math.isinf(value) else np.nan
        )
    return result
