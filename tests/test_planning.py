from pathlib import Path
from typing import Any

from sec_fin_rag.graph import ResearchGraph
from sec_fin_rag.manifest import load_issuers
from sec_fin_rag.models import ResearchPlan
from sec_fin_rag.planning import validate_research_plan

ISSUERS = load_issuers(Path(__file__).resolve().parents[1] / "data" / "manifest" / "issuers.json")


def test_explicit_range_overrides_incomplete_model_years() -> None:
    plan = validate_research_plan(
        {
            "tickers": ["GOOGL"],
            "metrics": ["revenue"],
            "years": [2023],
            "period_basis": "unspecified",
            "operation": "lookup",
        },
        "Compare Google's YoY revenue growth from 2023 to 2026",
        ISSUERS,
    )

    assert plan["years"] == [2023, 2024, 2025, 2026]
    assert plan["period_basis"] == "annual"
    assert plan["operation"] == "yoy"
    assert plan["tickers"] == ["GOOGL"]


def test_relative_year_count_is_not_replaced_by_model_guess() -> None:
    plan = validate_research_plan(
        {"tickers": ["GOOGL"], "years": [2024, 2025, 2026], "metrics": ["revenue"]},
        "Google revenue for the last three years",
        ISSUERS,
    )

    assert plan["years"] == []
    assert plan["last_n_completed_years"] == 3
    assert plan["period_basis"] == "annual"


def test_explicit_quarter_overrides_model_annual_guess() -> None:
    plan = validate_research_plan(
        {"tickers": ["GOOGL"], "metrics": ["revenue"], "period_basis": "annual"},
        "Compare Google Q2 revenue from 2023 to 2026",
        ISSUERS,
    )

    assert plan["period_basis"] == "quarterly"
    assert plan["quarter"] == 2
    assert plan["years"] == [2023, 2024, 2025, 2026]


def test_graph_plan_removes_single_year_filter_before_retrieval() -> None:
    observed: dict[str, Any] = {}

    def planner(_question: str, _previous: dict[str, Any]) -> ResearchPlan:
        return validate_research_plan(
            {"tickers": ["GOOGL"], "metrics": ["revenue"], "years": [2023]},
            "Compare Google YoY revenue growth from 2023 to 2026",
            ISSUERS,
        )

    def fundamentals(_question: str, _tickers: list[str], filters: dict[str, Any]) -> str:
        observed.update(filters)
        return ""

    graph = ResearchGraph(
        issuers=ISSUERS,
        retrieve=lambda *_args: (_ for _ in ()).throw(AssertionError("numeric range used RAG")),
        fundamentals=fundamentals,
        market=lambda _tickers: "",
        generate=lambda *_args: "",
        planner=planner,
    )
    result = graph.invoke(
        "Compare Google YoY revenue growth from 2023 to 2026",
        filters={"year": "All", "form": "All"},
    )

    assert result["route"] == "fundamentals"
    assert observed["year"] == "All"
    assert observed["form"] == "10-K"
    assert observed["_planned_years"] == [2023, 2024, 2025, 2026]
