import json
from pathlib import Path

from sec_fin_rag.models import Issuer
from sec_fin_rag.xbrl import _submission_report_dates, normalize_company_facts


def test_submission_report_dates_include_cached_archive_segments(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "submissions"
    cache.mkdir(parents=True)
    (cache / "CIK0001326801.json").write_text(
        json.dumps(
            {
                "filings": {
                    "recent": {
                        "accessionNumber": ["newer"],
                        "reportDate": ["2025-12-31"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (cache / "CIK0001326801-submissions-001.json").write_text(
        json.dumps({"accessionNumber": ["original"], "reportDate": ["2022-12-31"]}),
        encoding="utf-8",
    )

    assert _submission_report_dates(tmp_path, 1326801) == {
        "newer": "2025-12-31",
        "original": "2022-12-31",
    }


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


def test_company_facts_preserve_quarter_and_ytd_with_same_end_date(tmp_path: Path) -> None:
    rows = [
        {
            "val": value,
            "start": start,
            "end": "2025-06-30",
            "filed": filed,
            "form": "10-Q",
            "fy": fiscal_year,
            "fp": "Q2",
            "accn": accession,
        }
        for filed, fiscal_year, accession in (
            ("2025-07-31", 2025, "original"),
            ("2026-07-30", 2026, "comparative"),
        )
        for start, value in (
            ("2025-01-01", 89_830_000_000),
            ("2025-04-01", 47_516_000_000),
        )
    ]
    path = tmp_path / "facts.json"
    path.write_text(
        json.dumps({"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}}}}}),
        encoding="utf-8",
    )

    frame = normalize_company_facts(
        path,
        Issuer("Meta Platforms, Inc.", ("META",), 1326801, "Communication Services", "2026-09-30"),
    )

    assert len(frame) == 2
    by_period = frame.set_index("period_type")
    assert by_period.loc["three_months", "value"] == 47_516_000_000
    assert by_period.loc["six_months", "value"] == 89_830_000_000
    assert set(frame["accession"]) == {"original"}
    assert set(frame["fiscal_year"]) == {2025}  # Source filing FY, not the fact year.


def test_company_facts_prefer_matching_report_period_and_its_amendment(tmp_path: Path) -> None:
    rows = [
        {
            "val": value,
            "start": "2022-01-01",
            "end": "2022-12-31",
            "filed": filed,
            "form": form,
            "fy": fiscal_year,
            "fp": "FY",
            "accn": accession,
        }
        for accession, filed, fiscal_year, form, value in (
            ("original", "2023-02-02", 2022, "10-K", 116),
            ("amendment", "2023-03-02", 2022, "10-K/A", 117),
            ("later-comparative", "2025-01-30", 2024, "10-K", 116),
        )
    ]
    path = tmp_path / "facts.json"
    path.write_text(
        json.dumps({"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}}}}}),
        encoding="utf-8",
    )

    frame = normalize_company_facts(
        path,
        Issuer("Example", ("EX",), 1, "Technology", "2026-09-30"),
        {
            "original": "2022-12-31",
            "amendment": "2022-12-31",
            "later-comparative": "2024-12-31",
        },
    )

    assert len(frame) == 1
    assert frame.iloc[0]["accession"] == "amendment"
    assert frame.iloc[0]["value"] == 117
    assert frame.iloc[0]["source_report_end"] == "2022-12-31"
    assert bool(frame.iloc[0]["source_period_match"])
