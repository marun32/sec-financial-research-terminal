from __future__ import annotations

import os

# Tests must never emit LangSmith traces using credentials from a developer's .env.
os.environ["LANGSMITH_TRACING"] = "false"
# UI smoke tests must not make live Yahoo Finance requests.
os.environ["FINRAG_DISABLE_MARKET_AUTOLOAD"] = "true"
