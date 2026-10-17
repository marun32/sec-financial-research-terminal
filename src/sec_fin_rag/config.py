from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", "")
    nebius_base_url: str = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
    embedding_model: str = os.getenv("NEBIUS_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    chat_model: str = os.getenv("NEBIUS_CHAT_MODEL", "openai/gpt-oss-20b")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "sec-financial-rag")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "")

    @property
    def manifest_path(self) -> Path:
        return self.project_root / "data" / "manifest" / "issuers.json"

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def app_data_dir(self) -> Path:
        return self.project_root / "data" / "app"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports" / "generated"

    def ensure_local_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.app_data_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
