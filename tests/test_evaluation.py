import json
import math
from pathlib import Path

import pytest

from sec_fin_rag import evaluation
from sec_fin_rag.config import Settings
from sec_fin_rag.evaluation import (
    EvaluationConfig,
    EvidenceGroup,
    _evidence_groups,
    _matches_group,
    _metric_deltas,
    _metrics,
)
from sec_fin_rag.models import RetrievedChunk


def _chunk(*, ticker: str, text: str, section: str = "Item 1A. Risk Factors") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text=text,
        score=1.0,
        metadata={"ticker": ticker, "form": "10-K", "section": section},
    )


def test_evidence_match_requires_metadata_and_topical_text() -> None:
    group = EvidenceGroup(
        ticker="NVDA",
        forms=("10-K",),
        sections=("risk factors",),
        terms=("supply chain",),
    )
    assert _matches_group(_chunk(ticker="NVDA", text="Our supply chain is constrained."), group)
    assert not _matches_group(_chunk(ticker="MSFT", text="Our supply chain is constrained."), group)
    assert not _matches_group(_chunk(ticker="NVDA", text="Competition may increase."), group)


def test_evidence_section_anchor_can_appear_in_chunk_content() -> None:
    group = EvidenceGroup(
        ticker="META",
        forms=("10-K",),
        sections=("earnings per share",),
        terms=("diluted earnings per share",),
    )
    assert _matches_group(
        _chunk(
            ticker="META",
            section="Item 7. Management's Discussion and Analysis",
            text="Earnings per share results included diluted earnings per share of $23.49.",
        ),
        group,
    )


def test_single_company_metrics_are_normalized() -> None:
    metrics = _metrics([[frozenset(), frozenset({0})]], [1], [0.25])
    assert metrics.hit_at_5 == 1.0
    assert metrics.evidence_recall_at_5 == 1.0
    assert metrics.mrr_at_10 == 0.5
    assert metrics.ndcg_at_10 == pytest.approx(1 / math.log2(3))
    assert 0.0 <= metrics.ndcg_at_10 <= 1.0


def test_multi_company_requires_all_groups_and_ignores_duplicate_gain() -> None:
    ranking = [frozenset({0}), frozenset({0}), frozenset({1})]
    metrics = _metrics([ranking], [2], [0.5])
    expected_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics.hit_at_5 == 1.0
    assert metrics.evidence_recall_at_5 == 1.0
    assert metrics.mrr_at_10 == pytest.approx(1 / 3)
    assert metrics.ndcg_at_10 == pytest.approx(expected_ndcg)


def test_partial_multi_company_coverage_is_recall_but_not_hit() -> None:
    ranking = [frozenset({0}), frozenset({0})]
    metrics = _metrics([ranking], [2], [0.5])
    assert metrics.hit_at_5 == 0.0
    assert metrics.hit_at_10 == 0.0
    assert metrics.evidence_recall_at_5 == 0.5
    assert metrics.mrr_at_10 == 0.0


def test_unanswerable_has_no_retrieval_evidence_groups() -> None:
    assert (
        _evidence_groups(
            {"unanswerable": True, "expected_tickers": ["AAPL"], "expected_terms": ["forecast"]}
        )
        == ()
    )
    assert _metrics([], [], []) == _metrics([], [], [])


def test_metric_deltas_show_rerank_regressions() -> None:
    assert _metric_deltas(
        {"hit_at_10": 0.5, "ndcg_at_10": 0.8},
        {"hit_at_10": 0.25, "ndcg_at_10": 0.9},
    ) == {"hit_at_10": -0.25, "ndcg_at_10": pytest.approx(0.1)}


def test_metrics_reject_invalid_case_shapes() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        _metrics([[frozenset({0})]], [], [0.1])
    with pytest.raises(ValueError, match="at least one evidence group"):
        _metrics([[frozenset()]], [0], [0.1])


def test_retrieval_evaluate_does_not_run_answer_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path
    questions_dir = project_root / "data" / "evaluation"
    questions_dir.mkdir(parents=True)
    question = {
        "id": "numeric-01",
        "category": "numeric_table",
        "question": "What revenue did Apple report?",
        "expected_tickers": ["AAPL"],
        "expected_forms": ["10-K"],
        "expected_sections": ["financial statements"],
        "expected_terms": ["revenue"],
    }
    (questions_dir / "gold_questions.jsonl").write_text(json.dumps(question) + "\n")
    matching = _chunk(
        ticker="AAPL",
        section="Financial Statements",
        text="Revenue was reported for the fiscal year.",
    )

    monkeypatch.setattr(evaluation, "NebiusClient", lambda settings: object())
    monkeypatch.setattr(evaluation, "load_sparse_encoders", lambda path: object())
    monkeypatch.setattr(evaluation, "PineconeStore", lambda *args: object())
    monkeypatch.setattr(evaluation, "LocalReranker", lambda: object())
    monkeypatch.setattr(evaluation, "load_issuers", lambda path: [])
    monkeypatch.setattr(evaluation, "resolve_tickers", lambda question, issuers: ["AAPL"])
    monkeypatch.setattr(
        evaluation,
        "_evaluation_configs",
        lambda: [
            EvaluationConfig(name, name, 0.65)
            for name in ("fixed-v1", "semantic-v1", "ensemble", "auto")
        ],
    )
    monkeypatch.setattr(evaluation, "retrieve_candidates", lambda *args, **kwargs: [matching])
    monkeypatch.setattr(evaluation, "rerank_chunks", lambda *args, **kwargs: [matching])
    monkeypatch.setattr(
        evaluation,
        "evaluate_answers",
        lambda *args, **kwargs: pytest.fail("retrieval evaluation invoked answer evaluation"),
    )

    report = evaluation.evaluate(Settings(project_root=project_root, langsmith_api_key=""))

    assert "answer_evaluation" not in report
    assert (project_root / "reports" / "generated" / "retrieval_evaluation.json").exists()
    assert not (project_root / "reports" / "generated" / "answer_evaluation.json").exists()
