from __future__ import annotations

import json
from pathlib import Path

from .models import Issuer


def load_issuers(path: Path) -> list[Issuer]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot = payload["snapshot_date"]
    return [
        Issuer(
            name=item["name"],
            tickers=tuple(item["tickers"]),
            cik=int(item["cik"]),
            sector=item["sector"],
            snapshot_date=snapshot,
        )
        for item in payload["issuers"]
    ]


def ticker_map(issuers: list[Issuer]) -> dict[str, Issuer]:
    return {ticker.upper(): issuer for issuer in issuers for ticker in issuer.tickers}
