from __future__ import annotations

import re
from typing import Any

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(lsv2|pcsk|sk)-[A-Za-z0-9_-]{12,}\b"),
)
SECRET_FIELD = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.I)


def redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def sanitize_for_export(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SECRET_FIELD.search(str(key)) else sanitize_for_export(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_export(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_export(item) for item in value)
    return value


def redact(value: Any) -> Any:
    """Backward-compatible recursive redaction used by tracing and logging."""
    return sanitize_for_export(value)


def safe_error(error: Exception) -> str:
    return redact_text(str(error))[:500]
