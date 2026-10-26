from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import sec_fin_rag.ui as ui

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


def test_research_progress_collapses_and_remains_in_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(st, "page_link", lambda *args, **kwargs: None)

    class FakeGraph:
        def invoke(self, _question: str, **kwargs: object) -> dict[str, object]:
            callback = kwargs["on_step"]
            assert callable(callback)
            callback(
                "plan",
                {
                    "plan": {
                        "tickers": ["GOOGL"],
                        "years": [2023, 2024, 2025],
                        "last_n_completed_years": 0,
                    }
                },
            )
            callback(
                "route",
                {
                    "route": "fundamentals",
                    "filters": {"_planned_years": [2023, 2024, 2025]},
                },
            )
            callback("fundamentals", {"structured_evidence": [object()], "retrieved": []})
            callback("answer", {"citations": [object()], "citation_validation": {"is_valid": True}})
            return {
                "answer": "Revenue increased [Source 1].",
                "route": "fundamentals",
                "tickers": ["GOOGL"],
                "citations": [],
                "retrieved": [],
                "structured_evidence": [],
            }

    class FakeRuntime:
        graph = FakeGraph()

    monkeypatch.setattr(ui, "runtime", lambda: FakeRuntime())
    app = AppTest.from_file(str(ROOT / "pages/3_Research_Assistant.py"), default_timeout=15).run()
    app.chat_input[0].set_value("Google revenue for 2023 to 2025").run()

    assert not app.exception
    assert app.status[0].state == "complete"
    assert "complete" in app.status[0].label
    steps = app.session_state["research_messages"][-1]["research_steps"]
    assert len(steps) == 4
    assert "Evidence collected" in steps[2]

    app.run()
    assert not app.exception
    assert app.status[0].state == "complete"
