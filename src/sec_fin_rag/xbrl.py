from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .fact_periods import with_fact_periods
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


def _submission_report_dates(raw_dir: Path, cik: int) -> dict[str, str]:
    """Read cached SEC submission metadata; older archives may be only partly cached."""
    cache = raw_dir / "cache" / "submissions"
    stem = f"CIK{cik:010d}"
    paths = [cache / f"{stem}.json", *sorted(cache.glob(f"{stem}-submissions-*.json"))]
    report_dates: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("filings", {}).get("recent", {}) if "filings" in payload else payload
        for accession, report_date in zip(
            rows.get("accessionNumber", []), rows.get("reportDate", []), strict=False
        ):
            if accession and report_date:
                report_dates[str(accession)] = str(report_date)
    return report_dates


def normalize_company_facts(
    path: Path, issuer: Issuer, report_dates: Mapping[str, str] | None = None
) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report_dates = report_dates or {}
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
                    "source_report_end": report_dates.get(str(row.get("accn", ""))),
                    "concept": row.get("concept"),
                    "_concept_priority": row.get("_concept_priority", 0),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    # Preserve standalone quarters and YTD facts separately. For each fact period, prefer
    # the filing whose official report date matches that period; a matching amendment
    # supersedes its base filing. If older submission metadata is absent, the earliest
    # filing after the fact period is the closest available original-source proxy.
    frame["source_period_match"] = frame["source_report_end"].eq(frame["end"])
    amendments = frame["form"].str.endswith("/A")
    frame["_source_priority"] = 2
    frame.loc[frame["source_period_match"], "_source_priority"] = 1
    frame.loc[frame["source_period_match"] & amendments, "_source_priority"] = 0
    frame.loc[~frame["source_period_match"] & amendments, "_source_priority"] = 3
    frame["_base_form"] = frame["form"].str.removesuffix("/A")
    frame["_filing_order"] = pd.to_datetime(frame["filed"]).astype("int64")
    frame.loc[frame["_source_priority"] == 0, "_filing_order"] *= -1
    frame = frame.sort_values(
        ["_source_priority", "_filing_order", "_concept_priority"],
        ascending=[True, True, True],
        kind="stable",
    ).drop_duplicates(["ticker", "metric", "start", "end", "_base_form"], keep="first")
    return with_fact_periods(
        frame.drop(columns=["_concept_priority", "_source_priority", "_base_form", "_filing_order"])
    )


def build_fundamentals(issuers: list[Issuer], raw_dir: Path, output: Path) -> pd.DataFrame:
    frames = []
    for issuer in issuers:
        path = raw_dir / "companyfacts" / f"{issuer.cik:010d}.json"
        if path.exists():
            frames.append(
                normalize_company_facts(path, issuer, _submission_report_dates(raw_dir, issuer.cik))
            )
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result
