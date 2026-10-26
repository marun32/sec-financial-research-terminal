from sec_fin_rag.citations import (
    normalize_source_markers,
    number_citations,
    validate_answer_citations,
)
from sec_fin_rag.models import Citation
from sec_fin_rag.security import sanitize_for_export


def citation() -> Citation:
    return Citation("Apple", "10-K", "2026-01-01", "Risk Factors", "https://sec.gov", "c1")


def test_invalid_inline_source_markers_are_not_rendered_as_valid_citations() -> None:
    citations = number_citations([citation()])

    answer, validation = validate_answer_citations(
        "Supported [Source 1]. Bad [Source 9].", citations
    )

    assert "[Source 1]" in answer
    assert "[Source unavailable]" in answer
    assert validation["invalid_source_numbers"] == [9]
    assert validation["is_valid"] is False


def test_model_specific_citation_glyphs_are_normalized() -> None:
    assert normalize_source_markers("A【Source 3】 B【1†text】") == "A[Source 3] B[Source 1]"
    assert normalize_source_markers("Growth [Sources 1 & 2]") == ("Growth [Source 1] [Source 2]")


def test_exports_redact_secret_keys_and_token_patterns() -> None:
    payload = {
        "api_key": "secret",  # pragma: allowlist secret
        "message": "authorization: Bearer-value",  # pragma: allowlist secret
        "nested": ["lsv2-abcdefghijklmnop"],  # pragma: allowlist secret
    }

    sanitized = sanitize_for_export(payload)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["message"] == "[REDACTED]"
    assert sanitized["nested"] == ["[REDACTED]"]
