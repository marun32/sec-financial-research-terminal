from __future__ import annotations

import pandas as pd

from sec_fin_rag.models import Issuer
from sec_fin_rag.runtime import Runtime


def test_fundamental_evidence_filters_fact_period_by_requested_year() -> None:
    runtime = object.__new__(Runtime)
    runtime.issuers = [
        Issuer(
            name="Meta Platforms, Inc.",
            tickers=("META",),
            cik=1326801,
            sector="Communication Services",
            snapshot_date="2026-09-30",
        )
    ]
    runtime.fundamentals = pd.DataFrame(
        [
            {
                "ticker": "META",
                "metric": "revenue",
                "value": 200,
                "unit": "USD",
                "start": "2025-01-01",
                "end": "2025-12-31",
                "filed": "2026-01-29",
                "form": "10-K",
                "accession": "2025-filing",
            },
            {
                "ticker": "META",
                "metric": "revenue",
                "value": 120,
                "unit": "USD",
                "start": "2026-01-01",
                "end": "2026-06-30",
                "filed": "2026-07-30",
                "form": "10-Q",
                "accession": "2026-filing",
            },
        ]
    )

    evidence = runtime.fundamental_evidence(
        "META revenue in 2025", ["META"], {"year": "2025", "form": "All"}
    )

    assert len(evidence) == 1
    assert "2025-12-31" in evidence[0].text
    assert "2026-06-30" not in evidence[0].text
    assert '"fact_duration": "12 months"' in evidence[0].text


def test_fundamental_evidence_distinguishes_quarter_from_ytd() -> None:
    runtime = object.__new__(Runtime)
    runtime.issuers = [
        Issuer("Meta Platforms, Inc.", ("META",), 1326801, "Communication Services", "2026-09-30")
    ]
    runtime.fundamentals = pd.DataFrame(
        [
            {
                "ticker": "META",
                "metric": "revenue",
                "value": value,
                "unit": "USD",
                "start": start,
                "end": "2025-06-30",
                "filed": "2026-07-30",
                "form": "10-Q",
                "accession": "q2-comparative",
            }
            for start, value in (
                ("2025-01-01", 89_830_000_000),
                ("2025-04-01", 47_516_000_000),
            )
        ]
    )

    evidence = runtime.fundamental_evidence(
        "META Q2 2025 revenue", ["META"], {"year": "2025", "form": "10-Q"}
    )

    assert len(evidence) == 1
    assert '"fact_duration": "3 months"' in evidence[0].text
    assert '"fact_duration": "6 months"' in evidence[0].text
    assert '"start": "2025-04-01"' in evidence[0].text


def test_fundamental_evidence_cites_each_original_filing_separately() -> None:
    runtime = object.__new__(Runtime)
    runtime.issuers = [
        Issuer("Meta Platforms, Inc.", ("META",), 1326801, "Communication Services", "2026-09-30")
    ]
    runtime.fundamentals = pd.DataFrame(
        [
            {
                "ticker": "META",
                "metric": "revenue",
                "value": value,
                "unit": "USD",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
                "filed": filed,
                "form": "10-K",
                "accession": accession,
                "source_report_end": f"{year}-12-31",
                "source_period_match": True,
            }
            for year, value, filed, accession in (
                (2022, 116_609_000_000, "2023-02-02", "0001326801-23-000013"),
                (2023, 134_902_000_000, "2024-02-02", "0001326801-24-000012"),
            )
        ]
    )

    evidence = runtime.fundamental_evidence("META annual revenue", ["META"], {"form": "10-K"})

    assert len(evidence) == 2
    by_date = {item.citation.filing_date: item for item in evidence}
    assert by_date["2023-02-02"].citation.source_url.endswith(
        "/000132680123000013/0001326801-23-000013-index.htm"
    )
    assert '"end": "2022-12-31"' in by_date["2023-02-02"].text
    assert by_date["2024-02-02"].citation.source_url.endswith(
        "/000132680124000012/0001326801-24-000012-index.htm"
    )


def test_multi_year_revenue_evidence_keeps_annual_and_interim_comparisons_separate() -> None:
    runtime = object.__new__(Runtime)
    runtime.issuers = [
        Issuer("Alphabet Inc.", ("GOOGL", "GOOG"), 1652044, "Communication Services", "2026-09-30")
    ]
    rows = []
    for year, value, accession in (
        (2023, 307_394_000_000, "0001652044-24-000022"),
        (2024, 350_018_000_000, "0001652044-25-000014"),
        (2025, 402_836_000_000, "0001652044-26-000018"),
    ):
        rows.append(
            {
                "ticker": "GOOGL",
                "metric": "revenue",
                "value": value,
                "unit": "USD",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
                "form": "10-K",
                "filed": f"{year + 1}-02-05",
                "accession": accession,
                "source_report_end": f"{year}-12-31",
                "source_period_match": True,
                "fiscal_period": "FY",
            }
        )
    for year, value, accession in (
        (2025, 186_662_000_000, "0001652044-25-000062"),
        (2026, 229_692_000_000, "0001652044-26-000071"),
    ):
        rows.append(
            {
                "ticker": "GOOGL",
                "metric": "revenue",
                "value": value,
                "unit": "USD",
                "start": f"{year}-01-01",
                "end": f"{year}-06-30",
                "form": "10-Q",
                "filed": f"{year}-07-24",
                "accession": accession,
                "source_report_end": f"{year}-06-30",
                "source_period_match": True,
                "fiscal_period": "Q2",
            }
        )
    runtime.fundamentals = pd.DataFrame(rows)

    evidence = runtime.fundamental_evidence(
        "Google YoY revenue growth from 2023 to 2026",
        ["GOOGL"],
        {
            "form": "10-K",
            "year": "All",
            "_planned_years": [2023, 2024, 2025, 2026],
            "_period_basis": "annual",
            "_operation": "yoy",
        },
    )

    assert len(evidence) == 5
    text = " ".join(item.text for item in evidence)
    assert '"end": "2023-12-31"' in text
    assert '"end": "2024-12-31"' in text
    assert '"end": "2025-12-31"' in text
    assert '"end": "2025-06-30"' in text
    assert '"end": "2026-06-30"' in text
    assert "Annual facts not supplied: [2026]" in text


def test_multi_year_quarter_evidence_excludes_ytd_values() -> None:
    runtime = object.__new__(Runtime)
    runtime.issuers = [
        Issuer("Alphabet Inc.", ("GOOGL", "GOOG"), 1652044, "Communication Services", "2026-09-30")
    ]
    runtime.fundamentals = pd.DataFrame(
        [
            {
                "ticker": "GOOGL",
                "metric": "revenue",
                "value": value,
                "unit": "USD",
                "start": start,
                "end": f"{year}-06-30",
                "form": "10-Q",
                "filed": f"{year}-07-24",
                "accession": f"q2-{year}",
                "fiscal_period": "Q2",
            }
            for year in (2024, 2025)
            for start, value in ((f"{year}-04-01", 50), (f"{year}-01-01", 100))
        ]
    )

    evidence = runtime.fundamental_evidence(
        "Compare Google Q2 revenue from 2024 to 2025",
        ["GOOGL"],
        {"_planned_years": [2024, 2025], "_period_basis": "quarterly", "_quarter": 2},
    )

    assert len(evidence) == 2
    text = " ".join(item.text for item in evidence)
    assert text.count('"fact_duration": "3 months"') == 2
    assert '"fact_duration": "6 months"' not in text
