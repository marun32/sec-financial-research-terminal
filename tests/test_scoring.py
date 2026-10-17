import pandas as pd

from sec_fin_rag.scoring import score_companies


def test_scorecards_redistribute_missing_weights_and_report_coverage() -> None:
    rows = []
    for index, ticker in enumerate(("AAA", "BBB", "CCC"), start=1):
        rows.append(
            {
                "ticker": ticker,
                "revenue_cagr_3y": index / 10,
                "revenue_growth_yoy": index / 20,
                "eps_growth": None if ticker == "CCC" else index / 30,
                "operating_margin": index / 10,
                "net_margin": index / 12,
                "return_on_assets": index / 14,
                "fcf_margin": index / 16,
                "cash_to_debt": index,
                "inverse_debt_to_assets": -index / 10,
                "current_ratio": index / 2,
                "interest_coverage": index * 2,
            }
        )
    scored = score_companies(pd.DataFrame(rows))
    coverage = scored.loc[scored["ticker"] == "CCC", "Growth_coverage"].iloc[0]
    assert coverage == 0.7
    assert scored["Balanced"].notna().all()
