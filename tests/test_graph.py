from pathlib import Path
from typing import Any

from sec_fin_rag.graph import ResearchGraph, classify_route, extract_tickers
from sec_fin_rag.manifest import load_issuers
from sec_fin_rag.models import Citation, EvidenceItem, RetrievedChunk


def test_route_classification() -> None:
    assert classify_route("What is NVDA trading at today?") == "market"
    assert classify_route("What revenue did Apple report?") == "fundamentals"
    assert classify_route("Explain Microsoft's cybersecurity risks") == "filings"
    assert classify_route("How does today's price compare with reported revenue?") == "mixed"
    assert classify_route("Which stock should I buy with retirement savings?") == "unsupported"
    assert classify_route("What revenue was in Apple's latest annual filing?") == "mixed"
    assert classify_route("What net income was in Microsoft's 10-K?") == "mixed"


def test_ticker_extraction() -> None:
    assert extract_tickers("Compare MSFT with NVDA", {"MSFT", "NVDA", "AAPL"}) == ["MSFT", "NVDA"]


def test_graph_resolves_company_and_infers_form_before_retrieval() -> None:
    issuers = load_issuers(
        Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    )
    observed: dict[str, Any] = {}

    def retrieve(
        question: str, tickers: list[str], filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        observed.update(question=question, tickers=tickers, filters=filters)
        return []

    graph = ResearchGraph(
        issuers=issuers,
        retrieve=retrieve,
        fundamentals=lambda _question, _tickers, _filters: "",
        market=lambda _tickers: "",
        generate=lambda _question, _chunks, _context: "",
    )
    graph.invoke("Explain Apple's risks in its latest annual filing")

    assert observed["tickers"] == ["AAPL"]
    assert observed["filters"]["form"] == "10-K"


def test_structured_evidence_has_numbered_validated_provenance() -> None:
    issuers = load_issuers(
        Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    )
    evidence = EvidenceItem(
        text="Apple revenue was 100.",
        citation=Citation(
            issuer="Apple Inc.",
            form="SEC XBRL Company Facts",
            filing_date="2026-01-01",
            section="Structured fundamentals",
            source_url="https://data.sec.gov/example",
            chunk_id="xbrl:apple",
            source_type="xbrl",
        ),
    )
    graph = ResearchGraph(
        issuers=issuers,
        retrieve=lambda _question, _tickers, _filters: [],
        fundamentals=lambda _question, _tickers, _filters: [evidence],
        market=lambda _tickers: [],
        generate=lambda _question, _chunks, _evidence: "Revenue [Source 1], bad [Source 9].",
    )

    result = graph.invoke("What revenue did Apple report?")

    assert result["citations"] == []
    assert "could not produce an answer" in result["answer"]
    assert result["citation_validation"]["is_valid"] is False
    assert result["citation_validation"]["failed_closed"] is True


def test_graph_preserves_a_valid_cited_answer() -> None:
    issuers = load_issuers(
        Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    )
    chunk = RetrievedChunk(
        chunk_id="filing:1",
        text="Revenue was 100.",
        score=1.0,
        metadata={
            "issuer": "Apple Inc.",
            "form": "10-K",
            "filing_date": "2026-01-01",
            "section": "Financial statements",
            "source_url": "https://sec.gov/example",
        },
    )
    graph = ResearchGraph(
        issuers=issuers,
        retrieve=lambda _question, _tickers, _filters: [chunk],
        fundamentals=lambda _question, _tickers, _filters: [],
        market=lambda _tickers: [],
        generate=lambda _question, _chunks, _evidence: "Revenue was 100 [Source 1].",
    )

    result = graph.invoke("Explain Apple's revenue")

    assert result["citation_validation"]["is_valid"] is True
    assert result["citations"][0].source_number == 1


def test_fundamental_question_combines_xbrl_and_filing_retrieval() -> None:
    issuers = load_issuers(
        Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    )
    observed: dict[str, int] = {"retrieved": 0, "structured": 0}
    chunk = RetrievedChunk(
        chunk_id="filing:revenue",
        text="Management discussed revenue drivers.",
        score=1.0,
        metadata={
            "issuer": "Apple Inc.",
            "form": "10-K",
            "filing_date": "2026-01-01",
            "section": "MD&A",
            "source_url": "https://sec.gov/example",
        },
    )
    evidence = EvidenceItem(
        text="Revenue was 100.",
        citation=Citation(
            issuer="Apple Inc.",
            form="SEC XBRL Company Facts",
            filing_date="2026-01-01",
            section="Structured fundamentals",
            source_url="https://data.sec.gov/example",
            chunk_id="xbrl:apple",
            source_type="xbrl",
        ),
    )

    def generate(
        _question: str,
        chunks: list[RetrievedChunk],
        structured: list[EvidenceItem],
    ) -> str:
        observed["retrieved"] = len(chunks)
        observed["structured"] = len(structured)
        return "Revenue was 100 and management explained its drivers [Source 1] [Source 2]."

    graph = ResearchGraph(
        issuers=issuers,
        retrieve=lambda _question, _tickers, _filters: [chunk],
        fundamentals=lambda _question, _tickers, _filters: [evidence],
        market=lambda _tickers: [],
        generate=generate,
    )

    result = graph.invoke("What revenue did Apple report?")

    assert observed == {"retrieved": 1, "structured": 1}
    assert len(result["citations"]) == 2


def test_financial_overview_and_followup_reuse_thread_context() -> None:
    issuers = load_issuers(
        Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    )
    observed: list[tuple[str, list[str], dict[str, Any]]] = []

    def fundamentals(
        question: str, tickers: list[str], filters: dict[str, Any]
    ) -> list[EvidenceItem]:
        observed.append((question, tickers, filters))
        return []

    graph = ResearchGraph(
        issuers=issuers,
        retrieve=lambda _question, _tickers, _filters: [],
        fundamentals=fundamentals,
        market=lambda _tickers: [],
        generate=lambda _question, _chunks, _evidence: "",
    )

    first = graph.invoke(
        "Give me META financial information for 2026",
        filters={"form": "All", "year": "All", "namespace": "auto"},
        thread_id="meta-conversation",
    )
    followup = graph.invoke(
        "How about for 2025?",
        filters={"form": "All", "year": "All", "namespace": "auto"},
        thread_id="meta-conversation",
    )

    assert first["route"] == "fundamentals"
    assert first["tickers"] == ["META"]
    assert first["filters"]["year"] == "2026"
    assert followup["route"] == "fundamentals"
    assert followup["tickers"] == ["META"]
    assert followup["filters"]["year"] == "2025"
    assert "companies: META" in observed[-1][0]
    assert observed[-1][1] == ["META"]
    assert observed[-1][2]["year"] == "2025"


def test_conversation_context_does_not_leak_between_threads() -> None:
    issuers = load_issuers(
        Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    )
    graph = ResearchGraph(
        issuers=issuers,
        retrieve=lambda _question, _tickers, _filters: [],
        fundamentals=lambda _question, _tickers, _filters: [],
        market=lambda _tickers: [],
        generate=lambda _question, _chunks, _evidence: "",
    )

    graph.invoke("META revenue in 2026", thread_id="first-thread")
    result = graph.invoke("How about for 2025?", thread_id="second-thread")

    assert result["tickers"] == []
    assert result["route"] == "filings"
