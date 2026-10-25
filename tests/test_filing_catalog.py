from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sec_fin_rag.filing_catalog import build_filing_catalog


def test_catalog_uses_official_report_date_not_comparative_fact_date(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest = raw_dir / "filings" / "1045810" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            [
                {
                    "issuer": "NVIDIA Corporation",
                    "ticker": "NVDA",
                    "cik": 1045810,
                    "form": "10-Q",
                    "filing_date": "2026-08-26",
                    "report_date": "2026-07-26",
                    "accession": "0001045810-26-000075",
                    "source_url": "https://www.sec.gov/example",
                    "is_amendment": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    fundamentals = pd.DataFrame(
        [
            {
                "accession": "0001045810-26-000075",
                "end": "2025-07-27",
                "fiscal_year": 2027,
                "fiscal_period": "Q2",
            },
            {
                "accession": "0001045810-26-000075",
                "end": "2026-07-26",
                "fiscal_year": 2027,
                "fiscal_period": "Q2",
            },
        ]
    )

    catalog = build_filing_catalog(raw_dir, fundamentals, tmp_path / "filings.parquet")

    assert catalog.iloc[0]["report_date"] == "2026-07-26"
    assert catalog.iloc[0]["filing_date"] == "2026-08-26"
    assert catalog.iloc[0]["fiscal_year"] == 2027
    assert catalog.iloc[0]["fiscal_period"] == "Q2"
