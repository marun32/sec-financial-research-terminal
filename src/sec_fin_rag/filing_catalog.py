from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

CATALOG_COLUMNS = [
    "issuer",
    "ticker",
    "cik",
    "form",
    "filing_date",
    "report_date",
    "fiscal_year",
    "fiscal_period",
    "accession",
    "source_url",
    "is_amendment",
]


def sec_filing_index_url(cik: int, accession: str | None) -> str | None:
    """Link to the SEC filing-detail page for any Company Facts accession."""
    if (
        cik <= 0
        or not isinstance(accession, str)
        or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
    ):
        return None
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession.replace('-', '')}/{accession}-index.htm"
    )


def _period_labels(fundamentals: pd.DataFrame) -> pd.DataFrame:
    required = {"accession", "fiscal_year", "fiscal_period"}
    if fundamentals.empty or not required.issubset(fundamentals.columns):
        return pd.DataFrame(columns=["accession", "fiscal_year", "fiscal_period"])

    labels = fundamentals[["accession", "fiscal_year", "fiscal_period"]].dropna(
        subset=["accession"]
    )
    if labels.empty:
        return pd.DataFrame(columns=["accession", "fiscal_year", "fiscal_period"])

    # Company Facts repeats FY/FP on every fact context in a filing, including comparative
    # periods. Select the most frequently reported label per accession; the official report
    # date itself always comes from submissions metadata, never from these fact contexts.
    labels = (
        labels.value_counts(dropna=False)
        .rename("frequency")
        .reset_index()
        .sort_values(["accession", "frequency"], ascending=[True, False], kind="stable")
        .drop_duplicates("accession")
        .drop(columns="frequency")
    )
    labels["fiscal_year"] = pd.to_numeric(labels["fiscal_year"], errors="coerce").astype("Int64")
    return labels


def build_filing_catalog(raw_dir: Path, fundamentals: pd.DataFrame, output: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for manifest_path in sorted(raw_dir.glob("filings/*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for filing in payload:
            records.append(
                {
                    "issuer": filing["issuer"],
                    "ticker": filing["ticker"],
                    "cik": filing["cik"],
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "report_date": filing["report_date"],
                    "accession": filing["accession"],
                    "source_url": filing["source_url"],
                    "is_amendment": bool(filing.get("is_amendment", False)),
                }
            )

    catalog = pd.DataFrame.from_records(records)
    if catalog.empty:
        catalog = pd.DataFrame(columns=CATALOG_COLUMNS)
    else:
        catalog = catalog.drop_duplicates("accession")
        catalog = catalog.merge(_period_labels(fundamentals), on="accession", how="left")
        catalog = catalog[CATALOG_COLUMNS].sort_values(
            ["filing_date", "accession"], ascending=[False, False], kind="stable"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output, index=False)
    return catalog
