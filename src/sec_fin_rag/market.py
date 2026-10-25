from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _quote_value(info: Any, current_key: str, legacy_key: str) -> Any:
    """Read both current camel-case and legacy snake-case yfinance fields."""
    current = info.get(current_key)
    return current if current is not None else info.get(legacy_key)


def market_snapshot(tickers: list[str]) -> list[dict[str, Any]]:
    import yfinance as yf

    retrieved_at = datetime.now(UTC).isoformat()
    rows = []
    for ticker in tickers:
        quote = yf.Ticker(ticker)
        fast = quote.fast_info
        last = _quote_value(fast, "lastPrice", "last_price")
        previous = _quote_value(fast, "previousClose", "previous_close")
        rows.append(
            {
                "ticker": ticker,
                "price": last,
                "previous_close": previous,
                "change_percent": ((last / previous) - 1) * 100 if last and previous else None,
                "market_cap": _quote_value(fast, "marketCap", "market_cap"),
                "currency": _quote_value(fast, "currency", "currency"),
                "volume": _quote_value(fast, "lastVolume", "last_volume"),
                "exchange": _quote_value(fast, "exchange", "exchange"),
                "status": "available" if last is not None else "unavailable",
                "retrieved_at": retrieved_at,
                "source": "yfinance (may be delayed)",
            }
        )
    return rows
