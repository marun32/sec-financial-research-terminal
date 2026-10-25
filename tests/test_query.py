from pathlib import Path

from sec_fin_rag.manifest import load_issuers
from sec_fin_rag.query import (
    infer_filing_form,
    infer_financial_metrics,
    infer_year,
    pinecone_filter,
    resolve_tickers,
    with_inferred_form,
    with_inferred_year,
)

ROOT = Path(__file__).resolve().parents[1]
ISSUERS = load_issuers(ROOT / "data" / "manifest" / "issuers.json")


def test_resolves_company_names_aliases_and_share_classes() -> None:
    assert resolve_tickers("Compare Apple with Microsoft", ISSUERS) == ["AAPL", "MSFT"]
    assert resolve_tickers("Compare Google and Facebook", ISSUERS) == ["GOOGL", "META"]
    assert resolve_tickers("Compare GOOG with BRK.A", ISSUERS) == ["GOOGL", "BRK-B"]


def test_alias_matching_respects_word_boundaries() -> None:
    assert resolve_tickers("Discuss metadata quality", ISSUERS) == []


def test_infers_only_explicit_filing_period_language() -> None:
    assert infer_filing_form("What revenue appeared in Apple's latest annual filing?") == "10-K"
    assert infer_filing_form("What changed in Microsoft's quarterly report?") == "10-Q"
    assert infer_filing_form("What supply chain risks does NVIDIA identify?") is None
    assert infer_filing_form("Compare the available annual periods") == "10-K"


def test_explicit_filter_overrides_inferred_form() -> None:
    assert with_inferred_form("latest annual filing", {"form": "10-Q"})["form"] == "10-Q"
    assert with_inferred_form("latest annual filing", {"form": "All"})["form"] == "10-K"


def test_infers_reporting_year_and_financial_metrics() -> None:
    assert infer_year("How did META perform in 2025?") == "2025"
    assert infer_year("Show the latest META filing") is None
    assert with_inferred_year("META revenue in 2025", {"year": "All"})["year"] == "2025"
    assert with_inferred_year("META revenue in 2025", {"year": "2024"})["year"] == "2024"
    assert infer_financial_metrics("Compare revenue and net income") == [
        "revenue",
        "net_income",
    ]


def test_builds_pinecone_filter() -> None:
    assert pinecone_filter(["AAPL"], {"form": "10-K", "year": "2025"}) == {
        "ticker": {"$in": ["AAPL"]},
        "form": {"$in": ["10-K", "10-K/A"]},
        "report_year": {"$eq": 2025},
    }
