from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def market_snapshot(tickers: list[str]) -> list[dict[str, Any]]:
    import yfinance as yf

    retrieved_at = datetime.now(UTC).isoformat()
    rows = []
    for ticker in tickers:
        quote = yf.Ticker(ticker)
        fast = quote.fast_info
        last = fast.get("last_price")
        previous = fast.get("previous_close")
        rows.append(
            {
                "ticker": ticker,
                "price": last,
                "previous_close": previous,
                "change_percent": ((last / previous) - 1) * 100 if last and previous else None,
                "market_cap": fast.get("market_cap"),
                "retrieved_at": retrieved_at,
                "source": "yfinance (may be delayed)",
            }
        )
    return rows
