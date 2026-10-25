from __future__ import annotations

import pandas as pd

from sec_fin_rag.models import Issuer
from sec_fin_rag.runtime import Runtime


def test_fundamental_evidence_filters_fact_period_by_requested_year() -> None:
    runtime = object.__new__(Runtime)
    runtime.issuers = [
        Issuer(
            name="Meta Platforms, Inc.",
            tickers=("META",),
            cik=1326801,
            sector="Communication Services",
            snapshot_date="2026-09-30",
        )
    ]
    runtime.fundamentals = pd.DataFrame(
        [
            {
                "ticker": "META",
                "metric": "revenue",
                "value": 200,
                "unit": "USD",
                "end": "2025-12-31",
                "filed": "2026-01-29",
                "form": "10-K",
                "accession": "2025-filing",
            },
            {
                "ticker": "META",
                "metric": "revenue",
                "value": 120,
                "unit": "USD",
                "end": "2026-06-30",
                "filed": "2026-07-30",
                "form": "10-Q",
                "accession": "2026-filing",
            },
        ]
    )

    evidence = runtime.fundamental_evidence(
        "META revenue in 2025", ["META"], {"year": "2025", "form": "All"}
    )

    assert len(evidence) == 1
    assert "2025-12-31" in evidence[0].text
    assert "2026-06-30" not in evidence[0].text
