from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import Filing, Issuer


class SECClient:
    """Polite, cached SEC client capped below the published fair-access limit."""

    def __init__(self, user_agent: str, cache_dir: Path, requests_per_second: float = 5.0):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC_USER_AGENT must identify a person or organization and email address"
            )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._interval = 1.0 / requests_per_second
        self._last_request = 0.0
        self._lock = threading.Lock()
        self.client = httpx.Client(
            timeout=45,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/html,*/*",
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> SECClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _throttle(self) -> None:
        with self._lock:
            delay = self._interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()

    @retry(  # type: ignore[untyped-decorator]
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    def _fetch(self, url: str) -> bytes:
        self._throttle()
        response = self.client.get(url)
        response.raise_for_status()
        return cast(bytes, response.content)

    def cached_fetch(self, url: str, relative_path: Path) -> bytes:
        destination = self.cache_dir / relative_path
        if destination.exists() and destination.stat().st_size > 0:
            return destination.read_bytes()
        payload = cast(bytes, self._fetch(url))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return payload

    def get_json(self, url: str, relative_path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.cached_fetch(url, relative_path)))

    def submissions(self, issuer: Issuer) -> dict[str, Any]:
        cik = f"{issuer.cik:010d}"
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        return self.get_json(url, Path("submissions") / f"CIK{cik}.json")

    def submission_archive(self, filename: str) -> dict[str, Any]:
        url = f"https://data.sec.gov/submissions/{filename}"
        return self.get_json(url, Path("submissions") / filename)

    def company_facts(self, issuer: Issuer) -> dict[str, Any]:
        cik = f"{issuer.cik:010d}"
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        return self.get_json(url, Path("companyfacts") / f"CIK{cik}.json")

    @staticmethod
    def _merge_submission_rows(
        recent: dict[str, list[Any]], archive: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        recent_count = max((len(values) for values in recent.values()), default=0)
        archive_count = max((len(values) for values in archive.values()), default=0)
        keys = list(dict.fromkeys([*recent, *archive]))

        def padded(values: list[Any] | None, length: int) -> list[Any]:
            existing = list(values or [])
            return existing + [""] * (length - len(existing))

        return {
            key: [
                *padded(recent.get(key), recent_count),
                *padded(archive.get(key), archive_count),
            ]
            for key in keys
        }

    @staticmethod
    def _has_filing_targets(filings: list[Filing]) -> bool:
        return (
            sum(filing.form == "10-K" for filing in filings) >= 3
            and sum(filing.form == "10-Q" for filing in filings) >= 4
        )

    def complete_submissions(self, issuer: Issuer, submissions: dict[str, Any]) -> dict[str, Any]:
        """Add SEC archive segments only until the snapshot filing targets are satisfied."""
        filings = dict(submissions.get("filings", {}))
        recent = {key: list(values) for key, values in filings.get("recent", {}).items()}
        combined = {**submissions, "filings": {**filings, "recent": recent}}
        if self._has_filing_targets(self.select_filings(issuer, combined)):
            return combined

        archive_files = sorted(
            filings.get("files", []),
            key=lambda item: str(item.get("filingTo", "")),
            reverse=True,
        )
        for archive_metadata in archive_files:
            if str(archive_metadata.get("filingFrom", "")) > issuer.snapshot_date:
                continue
            filename = str(archive_metadata.get("name", ""))
            if not filename:
                continue
            archive = self.submission_archive(filename)
            recent = self._merge_submission_rows(recent, archive)
            combined = {**submissions, "filings": {**filings, "recent": recent}}
            if self._has_filing_targets(self.select_filings(issuer, combined)):
                break
        return combined

    def select_filings(self, issuer: Issuer, submissions: dict[str, Any]) -> list[Filing]:
        recent = submissions["filings"]["recent"]
        rows = [
            dict(zip(recent, values, strict=False))
            for values in zip(*recent.values(), strict=False)
        ]
        limits = {"10-K": 3, "10-Q": 4}
        eligible = [
            row
            for row in rows
            if str(row.get("form", "")).removesuffix("/A") in limits
            and bool(row.get("filingDate"))
            and str(row.get("filingDate", "")) <= issuer.snapshot_date
        ]
        eligible.sort(
            key=lambda row: (str(row.get("filingDate", "")), str(row.get("accessionNumber", ""))),
            reverse=True,
        )

        base_rows: list[dict[str, Any]] = []
        selected_periods: set[tuple[str, str]] = set()
        for base_form, limit in limits.items():
            selected_for_form = [row for row in eligible if str(row.get("form", "")) == base_form][
                :limit
            ]
            base_rows.extend(selected_for_form)
            selected_periods.update(
                (base_form, str(row.get("reportDate", ""))) for row in selected_for_form
            )

        amendment_rows = [
            row
            for row in eligible
            if str(row.get("form", "")).endswith("/A")
            and (
                str(row.get("form", "")).removesuffix("/A"),
                str(row.get("reportDate", "")),
            )
            in selected_periods
        ]
        selected_rows = sorted(
            base_rows + amendment_rows,
            key=lambda row: (str(row.get("filingDate", "")), str(row.get("accessionNumber", ""))),
            reverse=True,
        )

        selected: list[Filing] = []
        for row in selected_rows:
            form = str(row.get("form", ""))
            accession = str(row["accessionNumber"])
            primary = str(row["primaryDocument"])
            source_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{issuer.cik}/{accession.replace('-', '')}/{primary}"
            )
            selected.append(
                Filing(
                    issuer=issuer.name,
                    ticker=issuer.primary_ticker,
                    cik=issuer.cik,
                    form=form,
                    filing_date=str(row.get("filingDate", "")),
                    report_date=str(row.get("reportDate", "")),
                    accession=accession,
                    primary_document=primary,
                    source_url=source_url,
                    snapshot_date=issuer.snapshot_date,
                    is_amendment=form.endswith("/A"),
                )
            )
        return selected

    def download_issuer(self, issuer: Issuer, output_dir: Path) -> list[Filing]:
        submissions = self.complete_submissions(issuer, self.submissions(issuer))
        facts = self.company_facts(issuer)
        facts_path = output_dir / "companyfacts" / f"{issuer.cik:010d}.json"
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        facts_path.write_text(json.dumps(facts), encoding="utf-8")

        downloaded: list[Filing] = []
        for filing in self.select_filings(issuer, submissions):
            relative = (
                Path("filings") / str(issuer.cik) / filing.accession / filing.primary_document
            )
            payload = self.cached_fetch(filing.source_url, relative)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(payload)
            downloaded.append(Filing(**{**asdict(filing), "local_path": str(destination)}))

        manifest_path = output_dir / "filings" / str(issuer.cik) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps([asdict(filing) for filing in downloaded], indent=2), encoding="utf-8"
        )
        return downloaded
