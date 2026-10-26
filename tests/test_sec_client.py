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


def test_snapshot_cutoff_and_amendments_match_selected_periods(tmp_path: Path) -> None:
    rows = [
        ("10-K", "future", "2026-10-01", "2026-09-30"),
        ("10-K/A", "selected-amendment", "2026-09-15", "2025-12-31"),
        ("10-K", "selected-1", "2026-02-01", "2025-12-31"),
        ("10-K", "selected-2", "2025-02-01", "2024-12-31"),
        ("10-K", "selected-3", "2024-02-01", "2023-12-31"),
        ("10-K/A", "old-amendment", "2023-09-01", "2022-12-31"),
        ("10-K", "old-base", "2023-02-01", "2022-12-31"),
    ]
    recent = {
        "form": [row[0] for row in rows],
        "accessionNumber": [row[1] for row in rows],
        "filingDate": [row[2] for row in rows],
        "reportDate": [row[3] for row in rows],
        "primaryDocument": [f"{row[1]}.htm" for row in rows],
    }
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")
    with SECClient("Tester tester@example.com", tmp_path) as client:
        selected = client.select_filings(issuer, {"filings": {"recent": recent}})

    accessions = {item.accession for item in selected}
    assert accessions == {"selected-amendment", "selected-1", "selected-2", "selected-3"}
    assert all(item.filing_date <= issuer.snapshot_date for item in selected)


def test_archive_is_loaded_when_recent_submissions_do_not_meet_targets(
    tmp_path: Path, monkeypatch: object
) -> None:
    recent_rows = [
        ("10-K", "annual-1", "2026-02-01", "2025-12-31"),
        ("10-K", "annual-2", "2025-02-01", "2024-12-31"),
        *[("10-Q", f"quarter-{i}", f"2026-0{i + 3}-01", f"2026-0{i + 2}-28") for i in range(4)],
    ]
    recent = {
        "form": [row[0] for row in recent_rows],
        "accessionNumber": [row[1] for row in recent_rows],
        "filingDate": [row[2] for row in recent_rows],
        "reportDate": [row[3] for row in recent_rows],
        "primaryDocument": [f"{row[1]}.htm" for row in recent_rows],
    }
    archive = {
        "form": ["10-K"],
        "accessionNumber": ["annual-3"],
        "filingDate": ["2024-02-01"],
        "reportDate": ["2023-12-31"],
        "primaryDocument": ["annual-3.htm"],
    }
    submissions = {
        "filings": {
            "recent": recent,
            "files": [
                {
                    "name": "CIK0000000001-submissions-001.json",
                    "filingFrom": "2020-01-01",
                    "filingTo": "2024-06-01",
                }
            ],
        }
    }
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")
    with SECClient("Tester tester@example.com", tmp_path) as client:
        requested: list[str] = []

        def fake_archive(filename: str) -> dict[str, list[str]]:
            requested.append(filename)
            return archive

        monkeypatch.setattr(client, "submission_archive", fake_archive)  # type: ignore[attr-defined]
        completed = client.complete_submissions(issuer, submissions)
        selected = client.select_filings(issuer, completed)

    assert requested == ["CIK0000000001-submissions-001.json"]
    assert sum(item.form == "10-K" for item in selected) == 3
    assert sum(item.form == "10-Q" for item in selected) == 4


def test_archive_is_not_loaded_when_recent_submissions_are_complete(
    tmp_path: Path, monkeypatch: object
) -> None:
    forms = ["10-K"] * 3 + ["10-Q"] * 4
    recent = {
        "form": forms,
        "accessionNumber": [f"accession-{i}" for i in range(7)],
        "filingDate": ["2026-01-01"] * 7,
        "reportDate": [f"2025-0{i + 1}-01" for i in range(7)],
        "primaryDocument": [f"document-{i}.htm" for i in range(7)],
    }
    submissions = {
        "filings": {
            "recent": recent,
            "files": [{"name": "unused.json", "filingFrom": "2020-01-01"}],
        }
    }
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")
    with SECClient("Tester tester@example.com", tmp_path) as client:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            client,
            "submission_archive",
            lambda _: (_ for _ in ()).throw(AssertionError("archive should not be loaded")),
        )
        completed = client.complete_submissions(issuer, submissions)

    assert client._has_filing_targets(client.select_filings(issuer, completed))


def test_historical_submission_cache_fetches_metadata_not_future_segments(
    tmp_path: Path, monkeypatch: object
) -> None:
    issuer = Issuer("Example", ("EX",), 1, "Technology", "2026-09-30")
    submissions = {
        "filings": {
            "files": [
                {"name": "older.json", "filingFrom": "2010-01-01"},
                {"name": "future.json", "filingFrom": "2026-10-01"},
            ]
        }
    }
    with SECClient("Tester tester@example.com", tmp_path) as client:
        requested: list[str] = []
        monkeypatch.setattr(  # type: ignore[attr-defined]
            client, "submission_archive", lambda name: requested.append(name)
        )
        client.cache_submission_history(issuer, submissions)

    assert requested == ["older.json"]
