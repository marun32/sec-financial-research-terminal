from pathlib import Path

from sec_fin_rag.manifest import load_issuers, ticker_map


def test_manifest_has_ten_unique_issuers_and_consolidates_alphabet() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json"
    issuers = load_issuers(path)
    assert len(issuers) == 10
    assert len({issuer.cik for issuer in issuers}) == 10
    assert ticker_map(issuers)["GOOG"].cik == ticker_map(issuers)["GOOGL"].cik
