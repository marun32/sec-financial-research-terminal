from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import Issuer

CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "liabilities": ("Liabilities",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "debt": (
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebt",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtNoncurrent",
    ),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "interest_expense": ("InterestExpenseNonOperating", "InterestExpense"),
    "diluted_eps": ("EarningsPerShareDiluted",),
}


def _facts(payload: dict[str, Any], concepts: tuple[str, ...]) -> list[dict[str, Any]]:
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    records: list[dict[str, Any]] = []
    for priority, concept in enumerate(concepts):
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        preferred = "USD/shares" if "EarningsPerShare" in concept else "USD"
        if preferred in units:
            unit, rows = preferred, units[preferred]
        else:
            unit, rows = next(iter(units.items()), (preferred, []))
        records.extend(
            {
                **row,
                "concept": concept,
                "unit": unit,
                "_concept_priority": priority,
            }
            for row in rows
        )
    return records


def normalize_company_facts(path: Path, issuer: Issuer) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for metric, concepts in CONCEPTS.items():
        for row in _facts(payload, concepts):
            if row.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                continue
            filed = str(row.get("filed", ""))
            if not filed or filed > issuer.snapshot_date:
                continue
            records.append(
                {
                    "issuer": issuer.name,
                    "ticker": issuer.primary_ticker,
                    "cik": issuer.cik,
                    "sector": issuer.sector,
                    "snapshot_date": issuer.snapshot_date,
                    "metric": metric,
                    "value": row.get("val"),
                    "unit": row.get("unit", "USD"),
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "filed": filed,
                    "form": row.get("form"),
                    "fiscal_year": row.get("fy"),
                    "fiscal_period": row.get("fp"),
                    "accession": row.get("accn"),
                    "concept": row.get("concept"),
                    "_concept_priority": row.get("_concept_priority", 0),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    # Prefer the latest filing for a period and, within the same filing, the first concept in the
    # configured fallback order. Combining concepts prevents a stale historical tag from hiding
    # current facts reported under a successor tag.
    frame = frame.sort_values(
        ["filed", "_concept_priority"], ascending=[True, False], kind="stable"
    ).drop_duplicates(["ticker", "metric", "end", "form", "fiscal_period"], keep="last")
    return frame.drop(columns="_concept_priority")


def build_fundamentals(issuers: list[Issuer], raw_dir: Path, output: Path) -> pd.DataFrame:
    frames = []
    for issuer in issuers:
        path = raw_dir / "companyfacts" / f"{issuer.cik:010d}.json"
        if path.exists():
            frames.append(normalize_company_facts(path, issuer))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result
