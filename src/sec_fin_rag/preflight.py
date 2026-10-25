from __future__ import annotations

from dataclasses import dataclass

from langsmith import Client as LangSmithClient
from pinecone import Pinecone

from .config import Settings
from .providers import NebiusClient
from .security import safe_error


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    required: bool = True


def run_preflight(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    required = {
        "NEBIUS_API_KEY": settings.nebius_api_key,
        "PINECONE_API_KEY": settings.pinecone_api_key,
        "SEC_USER_AGENT": settings.sec_user_agent,
    }
    for name, value in required.items():
        valid = bool(value) and (name != "SEC_USER_AGENT" or "@" in value)
        checks.append(Check(name, valid, "configured" if valid else "missing or invalid"))

    if settings.nebius_api_key:
        try:
            NebiusClient(settings).validate_models()
            checks.append(Check("Nebius models", True, "embedding and chat models available"))
        except Exception as error:
            checks.append(Check("Nebius models", False, safe_error(error)))
    if settings.pinecone_api_key:
        try:
            Pinecone(api_key=settings.pinecone_api_key).list_indexes()
            checks.append(Check("Pinecone", True, "connection successful"))
        except Exception as error:
            checks.append(Check("Pinecone", False, safe_error(error)))
    if settings.langsmith_api_key:
        try:
            next(
                iter(LangSmithClient(api_key=settings.langsmith_api_key).list_datasets(limit=1)),
                None,
            )
            checks.append(Check("LangSmith", True, "optional connection successful", False))
        except Exception as error:
            checks.append(Check("LangSmith", False, f"optional: {safe_error(error)}", False))
    else:
        checks.append(Check("LangSmith", True, "optional; disabled", False))
    return checks
