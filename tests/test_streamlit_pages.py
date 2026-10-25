from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "streamlit_app.py",
    "pages/1_Company_Intelligence.py",
    "pages/2_Peer_Analysis.py",
    "pages/3_Research_Assistant.py",
    "pages/4_About_and_Data.py",
]


@pytest.mark.parametrize("relative_path", PAGES)
def test_streamlit_page_starts_without_errors(
    relative_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Local developer credentials must not cause smoke tests to emit external traces.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    if relative_path.startswith("pages/"):
        # An isolated page test has no multipage entrypoint, so navigation links cannot resolve.
        monkeypatch.setattr(st, "page_link", lambda *args, **kwargs: None)
    app = AppTest.from_file(str(ROOT / relative_path), default_timeout=15).run()

    assert not app.exception, f"{relative_path} failed to start: {app.exception}"
