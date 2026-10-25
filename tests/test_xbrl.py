import json
from pathlib import Path

from sec_fin_rag.models import Issuer
from sec_fin_rag.xbrl import normalize_company_facts


def test_company_facts_exclude_filings_after_snapshot(tmp_path: Path) -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "val": 100,
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "filed": "2026-02-01",
                                "form": "10-K",
                                "fy": 2025,
                                "fp": "FY",
                                "accn": "included",
                            },
                            {
                                "val": 110,
                                "start": "2026-01-01",
                                "end": "2026-09-30",
                                "filed": "2026-10-15",
                                "form": "10-Q",
                                "fy": 2026,
                                "fp": "Q3",
                                "accn": "excluded",
                            },
                        ]
                    }
                }
            }
        }
    }
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")

    frame = normalize_company_facts(path, issuer)

    assert frame["accession"].tolist() == ["included"]
    assert frame["snapshot_date"].tolist() == ["2026-09-30"]


def test_company_facts_uses_successor_concept_when_primary_tag_is_stale(tmp_path: Path) -> None:
    def fact(value: int, end: str, concept_year: int) -> dict[str, object]:
        return {
            "val": value,
            "start": f"{concept_year}-01-01",
            "end": end,
            "filed": f"{concept_year + 1}-02-01",
            "form": "10-K",
            "fy": concept_year,
            "fp": "FY",
            "accn": f"filing-{concept_year}",
        }

    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [fact(100, "2022-12-31", 2022)]}
                },
                "Revenues": {"units": {"USD": [fact(200, "2025-12-31", 2025)]}},
            }
        }
    }
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")

    frame = normalize_company_facts(path, issuer)

    assert frame["end"].tolist() == ["2022-12-31", "2025-12-31"]
    assert frame.iloc[-1]["concept"] == "Revenues"
    assert frame.iloc[-1]["value"] == 200
