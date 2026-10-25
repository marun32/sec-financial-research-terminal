from __future__ import annotations

import pandas as pd
import pytest

from sec_fin_rag.scoring import derive_company_metrics


def test_derived_metrics_align_by_period_end_not_restating_fiscal_year() -> None:
    rows = []
    for end, revenue, income in (
        ("2023-12-31", 100.0, 10.0),
        ("2024-12-31", 120.0, 18.0),
        ("2025-12-31", 150.0, 30.0),
    ):
        for metric, value in (
            ("revenue", revenue),
            ("net_income", income),
            ("operating_income", income * 1.2),
        ):
            rows.append(
                {
                    "ticker": "EX",
                    "issuer": "Example",
                    "sector": "Technology",
                    "metric": metric,
                    "value": value,
                    "end": end,
                    # Comparative values in a 2025 filing all inherit fy=2025 from SEC facts.
                    "fiscal_year": 2025,
                    "fiscal_period": "FY",
                    "form": "10-K",
                }
            )

    result = derive_company_metrics(pd.DataFrame(rows)).iloc[0]

    assert result["fiscal_year"] == 2025
    assert result["revenue_growth_yoy"] == pytest.approx(0.25)
    assert result["net_margin"] == pytest.approx(0.20)
