from __future__ import annotations

import hashlib
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from .models import DocumentBlock, Filing

_SPACE = re.compile(r"\s+")
_ITEM_HEADING = re.compile(
    r"^item\s+(?P<number>\d{1,2}[a-z]?)\s*[.\-—:]?\s*(?P<title>.*?)\s*$", re.IGNORECASE
)
_PART_HEADING = re.compile(
    r"^part\s+(?P<number>i{1,3}|iv|v)\b(?:\s*[.\-—:]?\s*(?P<title>.*?))?\s*$",
    re.IGNORECASE,
)
_ITEM_TITLES: dict[str, tuple[str, ...]] = {
    "1": ("Business", "Business Description", "Financial Statements", "Legal Proceedings"),
    "1A": ("Risk Factors",),
    "1B": ("Unresolved Staff Comments",),
    "1C": ("Cybersecurity",),
    "2": (
        "Properties",
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "Unregistered Sales of Equity Securities and Use of Proceeds",
    ),
    "3": (
        "Legal Proceedings",
        "Quantitative and Qualitative Disclosures About Market Risk",
        "Defaults Upon Senior Securities",
    ),
    "4": ("Mine Safety Disclosures", "Controls and Procedures"),
    "5": (
        "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases "
        "of Equity Securities",
        "Other Information",
    ),
    "6": ("Reserved", "Exhibits"),
    "7": ("Management's Discussion and Analysis of Financial Condition and Results of Operations",),
    "7A": ("Quantitative and Qualitative Disclosures About Market Risk",),
    "8": ("Financial Statements and Supplementary Data",),
    "9": ("Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",),
    "9A": ("Controls and Procedures",),
    "9B": ("Other Information",),
    "9C": ("Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",),
    "10": ("Directors, Executive Officers and Corporate Governance",),
    "11": ("Executive Compensation",),
    "12": ("Security Ownership of Certain Beneficial Owners and Management",),
    "13": ("Certain Relationships and Related Transactions",),
    "14": ("Principal Accountant Fees and Services",),
    "15": (
        "Exhibits and Financial Statement Schedules",
        "Exhibit and Financial Statement Schedules",
    ),
    "16": ("Form 10-K Summary",),
}
_TEN_K_DEFAULT_TITLES = {key: values[0] for key, values in _ITEM_TITLES.items()}
_TEN_Q_DEFAULT_TITLES: dict[tuple[str, str], str] = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): (
        "Management's Discussion and Analysis of Financial Condition and Results of Operations"
    ),
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "3"): "Defaults Upon Senior Securities",
    ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information",
    ("II", "6"): "Exhibits",
}


def clean_text(value: str) -> str:
    return _SPACE.sub(" ", value.replace("\xa0", " ")).strip()


def _normalized_heading(value: str) -> str:
    value = value.replace("’", "'").replace("&", "and")
    # SEC inline-XBRL generators sometimes split words across adjacent spans ("Ris" + "k").
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _part_heading(value: str) -> tuple[str, str] | None:
    match = _PART_HEADING.fullmatch(value)
    if not match:
        return None
    number = match.group("number").upper()
    title = clean_text(match.group("title") or "")
    return number, f"Part {number}" + (f". {title}" if title else "")


def _item_heading(value: str, form: str, part: str) -> str | None:
    match = _ITEM_HEADING.fullmatch(value)
    if not match:
        return None
    number = match.group("number").upper()
    title = clean_text(match.group("title")).strip(" .—-:")
    if re.search(r"\bitem\s+\d", title, re.IGNORECASE):
        return None

    canonical_title = ""
    normalized_title = _normalized_heading(title)
    for known_title in _ITEM_TITLES.get(number, ()):
        known_normalized = _normalized_heading(known_title)
        if normalized_title == known_normalized or (
            normalized_title.startswith(known_normalized)
            and normalized_title[len(known_normalized) :].strip().isdigit()
        ):
            canonical_title = known_title
            break
    if title and not canonical_title:
        return None
    if not canonical_title:
        base_form = form.removesuffix("/A")
        if base_form == "10-K":
            canonical_title = _TEN_K_DEFAULT_TITLES.get(number, "")
        elif base_form == "10-Q":
            canonical_title = _TEN_Q_DEFAULT_TITLES.get((part, number), "")
    suffix = f" {canonical_title}" if canonical_title else ""
    return f"Item {number}.{suffix}"


def _generic_heading(value: str, tag_name: str) -> bool:
    if not value or len(value) > 120 or sum(character.isalpha() for character in value) < 3:
        return False
    if tag_name in {"h1", "h2", "h3", "h4"}:
        return True
    return value.upper() == value and not set(value) <= {"_", "-", "—", "."}


def section_distribution_audit(blocks: list[DocumentBlock]) -> dict[str, Any]:
    by_filing: dict[str, Counter[str]] = {}
    filing_metadata: dict[str, tuple[str, str, str]] = {}
    for block in blocks:
        accession = str(block.metadata.get("accession", "unknown"))
        by_filing.setdefault(accession, Counter())[block.section] += 1
        filing_metadata[accession] = (
            str(block.metadata.get("ticker", "")),
            str(block.metadata.get("form", "")),
            str(block.metadata.get("filing_date", "")),
        )

    filings = []
    for accession, sections in sorted(by_filing.items()):
        ticker, form, filing_date = filing_metadata[accession]
        item_blocks = sum(
            count for section, count in sections.items() if section.startswith("Item ")
        )
        risk_blocks = sum(count for section, count in sections.items() if "Risk Factors" in section)
        filings.append(
            {
                "accession": accession,
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "block_count": sum(sections.values()),
                "item_labeled_blocks": item_blocks,
                "risk_factor_blocks": risk_blocks,
                "document_blocks": sections["Document"],
                "incorporated_reference_blocks": sections["DOCUMENTS INCORPORATED BY REFERENCE"],
                "sections": dict(sections.most_common()),
            }
        )
    periodic_filings = [
        row for row in filings if str(row["form"]).removesuffix("/A") in {"10-K", "10-Q"}
    ]
    missing_item_labels = [
        row["accession"] for row in periodic_filings if not row["item_labeled_blocks"]
    ]
    missing_risk_labels = [
        row["accession"]
        for row in periodic_filings
        if str(row["form"]) == "10-K" and not row["risk_factor_blocks"]
    ]
    return {
        "summary": {
            "filing_count": len(filings),
            "block_count": len(blocks),
            "item_labeled_block_count": sum(row["item_labeled_blocks"] for row in filings),
            "risk_factor_block_count": sum(row["risk_factor_blocks"] for row in filings),
            "incorporated_reference_block_count": sum(
                row["incorporated_reference_blocks"] for row in filings
            ),
            "missing_item_label_accessions": missing_item_labels,
            "missing_10k_risk_label_accessions": missing_risk_labels,
        },
        "filings": filings,
    }


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
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
        "snapshot_date": filing.snapshot_date,
        "accession": filing.accession,
        "source_url": filing.source_url,
    }
    blocks: list[DocumentBlock] = []
    seen: set[str] = set()
    section = "Document"
    primary_item_section = ""
    part = ""
    ordinal = 0
    root = soup.body or soup
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "div", "table"], recursive=True):
        if element.find_parent("table") and element.name != "table":
            continue
        if element.name == "div" and element.find(["div", "p", "table"], recursive=False):
            continue
        candidate = clean_text(element.get_text(" ", strip=True))
        part_match = _part_heading(candidate)
        if part_match:
            part, section = part_match
            primary_item_section = ""
            continue
        item_section = _item_heading(candidate, filing.form, part)
        if item_section:
            primary_item_section = item_section
            section = item_section
            continue
        if element.name != "table" and _generic_heading(candidate, element.name):
            section = (
                f"{primary_item_section} — {candidate[:120]}"
                if primary_item_section
                else candidate[:240]
            )
            continue
        text = table_to_markdown(element) if element.name == "table" else candidate
        if len(text) < 25:
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
