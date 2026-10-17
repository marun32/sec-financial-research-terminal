from pathlib import Path

from sec_fin_rag.models import Issuer
from sec_fin_rag.sec_client import SECClient


def test_selects_three_annual_and_four_quarterly_filings(tmp_path: Path) -> None:
    forms = ["10-K", "10-K", "10-K", "10-K", "10-Q", "10-Q", "10-Q", "10-Q", "10-Q"]
    recent = {
        "form": forms,
        "accessionNumber": [f"0001-26-00000{i}" for i in range(len(forms))],
        "primaryDocument": [f"doc{i}.htm" for i in range(len(forms))],
        "filingDate": ["2026-01-01"] * len(forms),
        "reportDate": ["2025-12-31"] * len(forms),
    }
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")
    with SECClient("Tester tester@example.com", tmp_path) as client:
        selected = client.select_filings(issuer, {"filings": {"recent": recent}})
    assert sum(item.form == "10-K" for item in selected) == 3
    assert sum(item.form == "10-Q" for item in selected) == 4
