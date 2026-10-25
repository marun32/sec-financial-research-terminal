from sec_fin_rag.config import Settings
from sec_fin_rag.preflight import run_preflight


def test_missing_langsmith_is_optional() -> None:
    checks = run_preflight(Settings(langsmith_api_key=""))
    langsmith = next(check for check in checks if check.name == "LangSmith")

    assert langsmith.passed
    assert not langsmith.required
    assert "optional" in langsmith.detail
