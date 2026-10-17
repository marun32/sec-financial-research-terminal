from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from .models import DocumentBlock, Filing

_SPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    return _SPACE.sub(" ", value.replace("\xa0", " ")).strip()


def table_to_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
            *("| " + " | ".join(row) + " |" for row in normalized[1:]),
        ]
    )


def parse_filing_html(html: str, filing: Filing) -> list[DocumentBlock]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    metadata: dict[str, Any] = {
        "issuer": filing.issuer,
        "ticker": filing.ticker,
        "cik": filing.cik,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "report_date": filing.report_date,
        "report_year": int(filing.report_date[:4]) if filing.report_date[:4].isdigit() else 0,
        "accession": filing.accession,
        "source_url": filing.source_url,
    }
    blocks: list[DocumentBlock] = []
    seen: set[str] = set()
    section = "Document"
    ordinal = 0
    root = soup.body or soup
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "div", "table"], recursive=True):
        if element.find_parent("table") and element.name != "table":
            continue
        if element.name == "div" and element.find(["div", "p", "table"], recursive=False):
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            candidate = clean_text(element.get_text(" ", strip=True))
            if candidate:
                section = candidate[:240]
            continue
        text = (
            table_to_markdown(element)
            if element.name == "table"
            else clean_text(element.get_text(" ", strip=True))
        )
        if len(text) < 25:
            continue
        if element.name != "table" and len(text) <= 240:
            normalized = text.strip()
            is_item_heading = bool(re.match(r"(?i)^item\s+\d+[a-z]?[.]?\s+", normalized))
            is_short_caps_heading = len(normalized) <= 120 and normalized.upper() == normalized
            if is_item_heading or is_short_caps_heading:
                section = normalized
                continue
        fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        blocks.append(
            DocumentBlock(
                block_id=f"{filing.accession}:{ordinal}:{fingerprint}",
                text=text,
                block_type="table" if element.name == "table" else "text",
                section=section,
                metadata=dict(metadata),
            )
        )
        ordinal += 1
    return blocks


def parse_filing_file(path: Path, filing: Filing) -> list[DocumentBlock]:
    return parse_filing_html(path.read_text(encoding="utf-8", errors="ignore"), filing)
