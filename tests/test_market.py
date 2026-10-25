from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sec_fin_rag.market import market_snapshot


def _install_yfinance(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    fake = SimpleNamespace(Ticker=lambda _ticker: SimpleNamespace(fast_info=payload))
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_market_snapshot_supports_current_fast_info_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_yfinance(
        monkeypatch,
        {
            "lastPrice": 105.0,
            "previousClose": 100.0,
            "marketCap": 2_000_000_000,
            "currency": "USD",
            "lastVolume": 50_000,
            "exchange": "NMS",
        },
    )

    row = market_snapshot(["TEST"])[0]

    assert row["price"] == 105.0
    assert row["change_percent"] == pytest.approx(5.0)
    assert row["market_cap"] == 2_000_000_000
    assert row["currency"] == "USD"
    assert row["volume"] == 50_000
    assert row["status"] == "available"


def test_market_snapshot_supports_legacy_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_yfinance(
        monkeypatch,
        {"last_price": 99.0, "previous_close": 100.0, "market_cap": 1_000_000},
    )

    row = market_snapshot(["TEST"])[0]

    assert row["price"] == 99.0
    assert row["change_percent"] == pytest.approx(-1.0)
    assert row["status"] == "available"


def test_market_snapshot_marks_incomplete_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_yfinance(monkeypatch, {})

    row = market_snapshot(["TEST"])[0]

    assert row["price"] is None
    assert row["change_percent"] is None
    assert row["status"] == "unavailable"
