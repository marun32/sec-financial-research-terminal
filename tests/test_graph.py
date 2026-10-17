from sec_fin_rag.graph import classify_route, extract_tickers


def test_route_classification() -> None:
    assert classify_route("What is NVDA trading at today?") == "market"
    assert classify_route("What revenue did Apple report?") == "fundamentals"
    assert classify_route("Explain Microsoft's cybersecurity risks") == "filings"
    assert classify_route("How does today's price compare with reported revenue?") == "mixed"
    assert classify_route("Which stock should I buy with retirement savings?") == "unsupported"


def test_ticker_extraction() -> None:
    assert extract_tickers("Compare MSFT with NVDA", {"MSFT", "NVDA", "AAPL"}) == ["MSFT", "NVDA"]
