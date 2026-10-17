from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_secret_files_are_gitignored() -> None:
    expected = [".env", ".env.local", ".streamlit/secrets.toml"]
    for relative in expected:
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"{relative} must be ignored by Git"


def test_env_example_contains_no_values() -> None:
    for line in (ROOT / ".env.example").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        assert key
        if key in {
            "NEBIUS_BASE_URL",
            "NEBIUS_EMBEDDING_MODEL",
            "NEBIUS_CHAT_MODEL",
            "PINECONE_INDEX",
            "LANGSMITH_PROJECT",
            "LANGSMITH_TRACING",
        }:
            continue
        if key == "SEC_USER_AGENT":
            assert value == "Your Name your.email@example.com"
        else:
            assert value == ""
