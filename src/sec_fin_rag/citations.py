from __future__ import annotations

import re
from dataclasses import replace

from .models import Citation

SOURCE_MARKER = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
DECORATED_SOURCE_MARKER = re.compile(r"【\s*(?:Source\s*)?(\d+)(?:\s*†[^】]*)?\s*】", re.IGNORECASE)


def normalize_source_markers(answer: str) -> str:
    """Normalize common model citation glyphs without inventing source numbers."""
    return DECORATED_SOURCE_MARKER.sub(lambda match: f"[Source {match.group(1)}]", answer)


def number_citations(citations: list[Citation]) -> list[Citation]:
    return [replace(citation, source_number=index) for index, citation in enumerate(citations, 1)]


def validate_answer_citations(
    answer: str, citations: list[Citation]
) -> tuple[str, dict[str, object]]:
    """Ensure every inline source marker points to a rendered citation."""
    answer = normalize_source_markers(answer)
    marker_numbers = [int(value) for value in SOURCE_MARKER.findall(answer)]
    valid_numbers = {citation.source_number for citation in citations}
    invalid_numbers = sorted(set(marker_numbers) - valid_numbers)

    def replace_invalid(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid_numbers else "[Source unavailable]"

    sanitized = SOURCE_MARKER.sub(replace_invalid, answer)
    validation: dict[str, object] = {
        "marker_count": len(marker_numbers),
        "valid_marker_count": sum(value in valid_numbers for value in marker_numbers),
        "invalid_source_numbers": invalid_numbers,
        "has_evidence": bool(citations),
        "has_valid_marker": any(value in valid_numbers for value in marker_numbers),
        "is_valid": not invalid_numbers and (not citations or bool(marker_numbers)),
    }
    return sanitized, validation
