from __future__ import annotations

import re
from typing import Any

_SECRET_FIELDS = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.I)


def redact(value: Any) -> Any:
    """Recursively remove secret-bearing fields before logging or tracing."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_FIELDS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def safe_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(api[_-]?key[=: ]+)[^\s,;]+", r"\1[REDACTED]", message)
    return message[:500]
