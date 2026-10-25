from __future__ import annotations

import csv
import json
import math
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith import traceable
from tqdm.auto import tqdm

from .config import Settings
from .manifest import load_issuers
from .models import GraphState, RetrievedChunk
from .providers import LocalReranker, NebiusClient
from .query import pinecone_filter, resolve_tickers, with_inferred_form
from .retrieval import (
    PineconeStore,
    rerank_chunks,
    retrieve_candidates,
    select_retrieval_strategy,
)
from .runtime import Runtime
from .security import safe_error, sanitize_for_export
from .sparse import load_sparse_encoders

ANSWER_EVALUATION_VERSION = "generation-v5"


@dataclass(frozen=True)
class EvidenceGroup:
    """One issuer-specific piece of evidence required to answer a gold question."""

    ticker: str
    forms: tuple[str, ...]
    sections: tuple[str, ...]
    terms: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_5: float
    hit_at_10: float
    evidence_recall_at_5: float
    evidence_recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    median_latency_seconds: float


@dataclass(frozen=True)
class EvaluationConfig:
    name: str
    strategy: str
    alpha: float
    candidate_count: int = 20
    top_n: int = 10


def _evidence_groups(item: dict[str, Any]) -> tuple[EvidenceGroup, ...]:
    if item.get("unanswerable"):
        return ()
    forms = tuple(str(value) for value in item.get("expected_forms", ()))
    sections = tuple(str(value).lower() for value in item.get("expected_sections", ()))
    terms = tuple(str(value).lower() for value in item.get("expected_terms", ()))
    return tuple(
        EvidenceGroup(ticker=str(ticker), forms=forms, sections=sections, terms=terms)
        for ticker in item.get("expected_tickers", ())
    )


def _matches_group(chunk: RetrievedChunk, group: EvidenceGroup) -> bool:
    ticker_ok = chunk.metadata.get("ticker") == group.ticker
    form = str(chunk.metadata.get("form", "")).removesuffix("/A")
    form_ok = not group.forms or form in group.forms
    section = str(chunk.metadata.get("section", "")).lower()
    text = chunk.text.lower()
    section_ok = not group.sections or any(
        value in section or value in text for value in group.sections
    )
    term_ok = not group.terms or any(value in text for value in group.terms)
    return ticker_ok and form_ok and section_ok and term_ok


def _ranked_group_matches(
    chunks: list[RetrievedChunk], groups: tuple[EvidenceGroup, ...]
) -> list[frozenset[int]]:
    return [
        frozenset(index for index, group in enumerate(groups) if _matches_group(chunk, group))
        for chunk in chunks
    ]


def _covered_groups(matches: list[frozenset[int]], k: int) -> set[int]:
    covered: set[int] = set()
    for group_matches in matches[:k]:
        covered.update(group_matches)
    return covered


def _metrics(
    rankings: list[list[frozenset[int]]],
    group_counts: list[int],
    latencies: list[float],
) -> RetrievalMetrics:
    if not (len(rankings) == len(group_counts) == len(latencies)):
        raise ValueError("rankings, group counts, and latencies must have equal lengths")
    if any(count <= 0 for count in group_counts):
        raise ValueError("answerable questions must contain at least one evidence group")
    if not rankings:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def hit_at(values: list[frozenset[int]], count: int, k: int) -> float:
        return float(len(_covered_groups(values, k)) == count)

    def evidence_recall_at(values: list[frozenset[int]], count: int, k: int) -> float:
        return len(_covered_groups(values, k)) / count

    def reciprocal_rank(values: list[frozenset[int]], count: int, k: int) -> float:
        covered: set[int] = set()
        for rank, group_matches in enumerate(values[:k], start=1):
            covered.update(group_matches)
            if len(covered) == count:
                return 1 / rank
        return 0.0

    def ndcg(values: list[frozenset[int]], count: int, k: int) -> float:
        covered: set[int] = set()
        dcg = 0.0
        for rank, group_matches in enumerate(values[:k], start=1):
            new_groups = group_matches - covered
            dcg += len(new_groups) / math.log2(rank + 1)
            covered.update(new_groups)
        ideal_hits = min(count, k)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / ideal_dcg if ideal_dcg else 0.0

    cases = list(zip(rankings, group_counts, strict=True))
    return RetrievalMetrics(
        hit_at_5=statistics.fmean(hit_at(row, count, 5) for row, count in cases),
        hit_at_10=statistics.fmean(hit_at(row, count, 10) for row, count in cases),
        evidence_recall_at_5=statistics.fmean(
            evidence_recall_at(row, count, 5) for row, count in cases
        ),
        evidence_recall_at_10=statistics.fmean(
            evidence_recall_at(row, count, 10) for row, count in cases
        ),
        mrr_at_10=statistics.fmean(reciprocal_rank(row, count, 10) for row, count in cases),
        ndcg_at_10=statistics.fmean(ndcg(row, count, 10) for row, count in cases),
        median_latency_seconds=statistics.median(latencies),
    )


def _category_metrics(
    rankings: list[list[frozenset[int]]],
    group_counts: list[int],
    latencies: list[float],
    categories: list[str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for category in sorted(set(categories)):
        indexes = [index for index, value in enumerate(categories) if value == category]
        result[category] = asdict(
            _metrics(
                [rankings[index] for index in indexes],
                [group_counts[index] for index in indexes],
                [latencies[index] for index in indexes],
            )
        )
    return result


def _metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - value for key, value in before.items()}


def _group_ranks(
    matches: list[frozenset[int]], groups: tuple[EvidenceGroup, ...]
) -> dict[str, list[int]]:
    return {
        group.ticker: [rank for rank, values in enumerate(matches, start=1) if index in values]
        for index, group in enumerate(groups)
    }


def _evaluation_configs() -> list[EvaluationConfig]:
    configurations: list[EvaluationConfig] = []
    for namespace in ("fixed-v1", "semantic-v1"):
        for alpha in (0.0, 0.35, 0.5, 0.65, 0.8, 1.0):
            name = namespace if alpha == 0.65 else f"{namespace}-alpha-{alpha:.2f}"
            configurations.append(EvaluationConfig(name, namespace, alpha))
    configurations.extend(
        [
            EvaluationConfig("ensemble", "ensemble", 0.65),
            EvaluationConfig("auto", "auto", 0.65),
            EvaluationConfig("ensemble-candidates-20-top-8", "ensemble", 0.65, 20, 8),
            EvaluationConfig("ensemble-candidates-30-top-10", "ensemble", 0.65, 30, 10),
        ]
    )
    return configurations


def _refusal_detected(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in (
            "cannot provide",
            "can't provide",
            "cannot predict",
            "can't predict",
            "could not find sufficient evidence",
            "confidential",
            "not available",
        )
    )


def _write_answer_checkpoint(
    path: Any, rows_by_id: dict[str, dict[str, Any]], questions: list[dict[str, Any]]
) -> None:
    ordered = [rows_by_id[str(item["id"])] for item in questions if str(item["id"]) in rows_by_id]
    path.write_text(
        "".join(json.dumps(sanitize_for_export(row), ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )


def _answer_evidence(result: GraphState) -> list[dict[str, Any]]:
    evidence = [
        {
            "source": index,
            "text": chunk.text,
            "issuer": chunk.metadata.get("issuer"),
            "form": chunk.metadata.get("form"),
            "filing_date": chunk.metadata.get("filing_date"),
            "section": chunk.metadata.get("section"),
        }
        for index, chunk in enumerate(result.get("retrieved", []), start=1)
    ]
    offset = len(evidence)
    evidence.extend(
        {
            "source": offset + index,
            "text": item.text,
            "issuer": item.citation.issuer,
            "form": item.citation.form,
            "filing_date": item.citation.filing_date,
            "section": item.citation.section,
        }
        for index, item in enumerate(result.get("structured_evidence", []), start=1)
    )
    return evidence


@traceable(  # type: ignore[untyped-decorator]
    name="evaluate_answer_case",
    run_type="chain",
    process_inputs=lambda inputs: {"item": inputs.get("item", {})},
)
def _evaluate_answer_case(
    runtime: Runtime, item: dict[str, Any]
) -> tuple[GraphState, dict[str, Any]]:
    result = runtime.graph.invoke(
        str(item["question"]),
        filters={"form": "All", "year": "All", "namespace": "auto"},
    )
    if item.get("unanswerable"):
        return result, {}
    judgment = runtime.nebius.judge_answer(
        question=str(item["question"]),
        answer=str(result.get("answer", "")),
        evidence=_answer_evidence(result),
    )
    return result, judgment


def evaluate_answers(
    settings: Settings, questions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if questions is None:
        questions_path = settings.project_root / "data" / "evaluation" / "gold_questions.jsonl"
        questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line]
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = settings.reports_dir / "answer_evaluation_rows.jsonl"
    rows_by_id: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_by_id[str(row.get("id", ""))] = row
    completed_ids = {
        row_id
        for row_id, row in rows_by_id.items()
        if row.get("status") == "complete"
        and row.get("evaluation_version") == ANSWER_EVALUATION_VERSION
    }
    runtime = Runtime(settings)
    progress = tqdm(total=len(questions), initial=len(completed_ids), desc="Answer evaluation")
    for item in questions:
        item_id = str(item["id"])
        if item_id in completed_ids:
            continue
        started = time.perf_counter()
        try:
            result, judgment = _evaluate_answer_case(runtime, item)
            validation = result.get("citation_validation", {})
            unanswerable = bool(item.get("unanswerable"))
            row = {
                "id": item_id,
                "evaluation_version": ANSWER_EVALUATION_VERSION,
                "status": "complete",
                "category": item["category"],
                "question": item["question"],
                "answer": result.get("answer", ""),
                "source_count": len(result.get("citations", [])),
                "citation_markers_valid_auto": validation.get("is_valid"),
                "valid_marker_count": validation.get("valid_marker_count", 0),
                "invalid_source_numbers": validation.get("invalid_source_numbers", []),
                "refusal_correct_auto": _refusal_detected(str(result.get("answer", "")))
                if unanswerable
                else "NA",
                "faithfulness": judgment.get("faithfulness", {}).get("score", "NA"),
                "faithfulness_reason": judgment.get("faithfulness", {}).get("reason", ""),
                "answer_relevancy": judgment.get("answer_relevancy", {}).get("score", "NA"),
                "answer_relevancy_reason": judgment.get("answer_relevancy", {}).get("reason", ""),
                "citation_correctness": judgment.get("citation_correctness", {}).get("score", "NA"),
                "citation_correctness_reason": judgment.get("citation_correctness", {}).get(
                    "reason", ""
                ),
                "citation_completeness": judgment.get("citation_completeness", {}).get(
                    "score", "NA"
                ),
                "citation_completeness_reason": judgment.get("citation_completeness", {}).get(
                    "reason", ""
                ),
                "latency_seconds": time.perf_counter() - started,
                "error": "",
                "faithful_0_or_1": "",
                "citations_correct_0_or_1": "",
                "numeric_accuracy_0_or_1_or_na": "",
                "refusal_correct_0_or_1_or_na": "",
                "review_notes": "",
            }
        except Exception as error:
            row = {
                "id": item_id,
                "evaluation_version": ANSWER_EVALUATION_VERSION,
                "status": "failed",
                "category": item["category"],
                "question": item["question"],
                "answer": "",
                "source_count": 0,
                "citation_markers_valid_auto": False,
                "valid_marker_count": 0,
                "invalid_source_numbers": [],
                "refusal_correct_auto": False if item.get("unanswerable") else "NA",
                "faithfulness": "NA",
                "faithfulness_reason": "",
                "answer_relevancy": "NA",
                "answer_relevancy_reason": "",
                "citation_correctness": "NA",
                "citation_correctness_reason": "",
                "citation_completeness": "NA",
                "citation_completeness_reason": "",
                "latency_seconds": time.perf_counter() - started,
                "error": safe_error(error),
                "faithful_0_or_1": "",
                "citations_correct_0_or_1": "",
                "numeric_accuracy_0_or_1_or_na": "",
                "refusal_correct_0_or_1_or_na": "",
                "review_notes": "",
            }
        rows_by_id[item_id] = row
        _write_answer_checkpoint(checkpoint_path, rows_by_id, questions)
        progress.update(1)
    progress.close()
    rows = [rows_by_id[str(item["id"])] for item in questions if str(item["id"]) in rows_by_id]
    sanitized_rows = sanitize_for_export(rows)
    review_path = settings.reports_dir / "answer_review.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sanitized_rows)
    question_by_id = {str(item["id"]): item for item in questions}
    complete = [row for row in rows if row.get("status") == "complete"]
    answerable = [row for row in complete if not question_by_id[str(row["id"])].get("unanswerable")]
    refusals = [row for row in complete if question_by_id[str(row["id"])].get("unanswerable")]
    summary = {
        "answer_count": len(complete),
        "failed_question_ids": [row["id"] for row in rows if row.get("status") != "complete"],
        "citation_marker_integrity": statistics.fmean(
            bool(row["citation_markers_valid_auto"]) for row in answerable
        )
        if answerable
        else 0.0,
        "refusal_correctness_auto": statistics.fmean(
            bool(row["refusal_correct_auto"]) for row in refusals
        )
        if refusals
        else 0.0,
        "faithfulness": statistics.fmean(float(row["faithfulness"]) for row in answerable)
        if answerable
        else 0.0,
        "answer_relevancy": statistics.fmean(float(row["answer_relevancy"]) for row in answerable)
        if answerable
        else 0.0,
        "citation_correctness": statistics.fmean(
            float(row["citation_correctness"]) for row in answerable
        )
        if answerable
        else 0.0,
        "citation_completeness": statistics.fmean(
            float(row["citation_completeness"]) for row in answerable
        )
        if answerable
        else 0.0,
        "median_end_to_end_latency_seconds": statistics.median(
            float(row["latency_seconds"]) for row in complete
        )
        if complete
        else 0.0,
        "manual_review_required": True,
        "review_path": str(review_path),
        "checkpoint_path": str(checkpoint_path),
    }
    (settings.reports_dir / "answer_evaluation.json").write_text(
        json.dumps(sanitize_for_export(summary), indent=2), encoding="utf-8"
    )
    return summary


def evaluate_with_langsmith(settings: Settings) -> Any:
    """Run generation metrics as a named LangSmith dataset experiment."""
    if not settings.langsmith_api_key:
        raise ValueError(
            "LANGSMITH_API_KEY is not configured; LangSmith export is an optional command"
        )
    questions_path = settings.project_root / "data" / "evaluation" / "gold_questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line]
    client = LangSmithClient(api_key=settings.langsmith_api_key)
    dataset_name = "sec-financial-rag-gold-v1"
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name,
            description="SEC financial research gold questions for the frozen project snapshot.",
        )
    namespace = uuid.UUID("af914e85-4324-41c0-9e46-77a0c192fe58")
    examples = [
        {
            "id": str(uuid.uuid5(namespace, str(item["id"]))),
            "inputs": {
                "id": item["id"],
                "question": item["question"],
                "unanswerable": bool(item.get("unanswerable")),
            },
            "outputs": {"expected_refusal": bool(item.get("unanswerable"))},
        }
        for item in questions
    ]
    existing_ids = {
        str(example.id)
        for example in client.list_examples(dataset_id=dataset.id, include_attachments=False)
    }
    existing = [example for example in examples if str(example["id"]) in existing_ids]
    missing = [example for example in examples if str(example["id"]) not in existing_ids]
    if existing:
        client.update_examples(dataset_id=dataset.id, updates=existing)
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)
    runtime = Runtime(settings)
    cached_rows: dict[str, dict[str, Any]] = {}
    checkpoint_path = settings.reports_dir / "answer_evaluation_rows.jsonl"
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("status") == "complete"
                and row.get("evaluation_version") == ANSWER_EVALUATION_VERSION
            ):
                cached_rows[str(row.get("id", ""))] = row

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        cached = cached_rows.get(str(inputs["id"]))
        if cached is not None:
            return {
                "answer": cached.get("answer", ""),
                "citation_marker_integrity": int(bool(cached.get("citation_markers_valid_auto"))),
                "refusal_correctness": (
                    int(bool(cached.get("refusal_correct_auto")))
                    if inputs.get("unanswerable")
                    else None
                ),
                "faithfulness": cached.get("faithfulness"),
                "answer_relevancy": cached.get("answer_relevancy"),
                "citation_correctness": cached.get("citation_correctness"),
                "citation_completeness": cached.get("citation_completeness"),
            }
        item = {
            "id": inputs["id"],
            "question": inputs["question"],
            "unanswerable": inputs.get("unanswerable", False),
        }
        result, judgment = _evaluate_answer_case(runtime, item)
        return {
            "answer": result.get("answer", ""),
            "citation_marker_integrity": int(
                bool(result.get("citation_validation", {}).get("is_valid"))
            ),
            "refusal_correctness": (
                int(_refusal_detected(str(result.get("answer", ""))))
                if item["unanswerable"]
                else None
            ),
            **{name: value.get("score") for name, value in judgment.items()},
        }

    def score(name: str) -> Any:
        def evaluator(outputs: dict[str, Any]) -> dict[str, Any]:
            value = outputs.get(name)
            if value in {None, "", "NA"}:
                return {"key": name, "score": None, "comment": "Not applicable"}
            return {"key": name, "score": float(value)}

        evaluator.__name__ = name
        return evaluator

    return client.evaluate(
        target,
        data=dataset_name,
        evaluators=[
            score("citation_marker_integrity"),
            score("refusal_correctness"),
            score("faithfulness"),
            score("answer_relevancy"),
            score("citation_correctness"),
            score("citation_completeness"),
        ],
        experiment_prefix="sec-financial-rag-generation-v5",
        description="Focused generation evaluation with LLM and deterministic graders.",
        metadata={"evaluation_version": ANSWER_EVALUATION_VERSION},
        max_concurrency=1,
    )


def evaluate(settings: Settings) -> dict[str, Any]:
    questions_path = settings.project_root / "data" / "evaluation" / "gold_questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line]
    nebius = NebiusClient(settings)
    sparse = load_sparse_encoders(settings.app_data_dir)
    store = PineconeStore(settings, nebius, sparse)
    reranker = LocalReranker()
    issuers = load_issuers(settings.manifest_path)
    configurations = _evaluation_configs()
    report: dict[str, Any] = {
        "question_count": len(questions),
        "answerable_question_count": sum(not item.get("unanswerable") for item in questions),
        "metric_definitions": {
            "hit_at_k": "fraction of questions with every required evidence group covered by k",
            "evidence_recall_at_k": "mean fraction of required evidence groups covered by k",
            "mrr_at_10": "reciprocal rank where all required evidence groups are first covered",
            "ndcg_at_10": "group-aware normalized DCG; duplicate group hits receive no gain",
        },
        "configurations": {},
    }
    for configuration in configurations:
        before: list[list[frozenset[int]]] = []
        after: list[list[frozenset[int]]] = []
        before_latency: list[float] = []
        after_latency: list[float] = []
        group_counts: list[int] = []
        categories: list[str] = []
        details = []
        regression_ids: list[str] = []
        for item in questions:
            tickers = resolve_tickers(item["question"], issuers)
            filters = with_inferred_form(item["question"])
            query_filter = pinecone_filter(tickers, filters)
            started = time.perf_counter()
            active_strategy = configuration.strategy
            if active_strategy == "auto":
                active_strategy = select_retrieval_strategy(item["question"], tickers).strategy
            retrieved = retrieve_candidates(
                store,
                item["question"],
                active_strategy,
                query_filter,
                top_k=30,
                alpha=configuration.alpha,
            )
            retrieval_latency = time.perf_counter() - started
            started = time.perf_counter()
            reranked = rerank_chunks(
                item["question"],
                retrieved,
                reranker,
                candidate_count=configuration.candidate_count,
                top_n=configuration.top_n,
                preserve_tickers=tickers,
            )
            rerank_latency = retrieval_latency + time.perf_counter() - started
            groups = _evidence_groups(item)
            before_matches = _ranked_group_matches(retrieved, groups)
            after_matches = _ranked_group_matches(reranked, groups)
            if groups:
                before.append(before_matches)
                after.append(after_matches)
                before_latency.append(retrieval_latency)
                after_latency.append(rerank_latency)
                group_counts.append(len(groups))
                categories.append(str(item["category"]))
                before_complete = len(_covered_groups(before_matches, 10)) == len(groups)
                after_complete = len(_covered_groups(after_matches, 10)) == len(groups)
                if before_complete and not after_complete:
                    regression_ids.append(str(item["id"]))
            else:
                before_complete = False
                after_complete = False
            details.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "unanswerable": bool(item.get("unanswerable")),
                    "resolved_tickers": tickers,
                    "resolved_form": filters.get("form"),
                    "selected_strategy": active_strategy,
                    "required_evidence_tickers": [group.ticker for group in groups],
                    "before_group_ranks": _group_ranks(before_matches, groups),
                    "after_group_ranks": _group_ranks(after_matches, groups),
                    "before_complete_at_10": before_complete,
                    "after_complete_at_10": after_complete,
                    "rerank_regression_at_10": before_complete and not after_complete,
                }
            )
        before_metrics = asdict(_metrics(before, group_counts, before_latency))
        after_metrics = asdict(_metrics(after, group_counts, after_latency))
        before_categories = _category_metrics(before, group_counts, before_latency, categories)
        after_categories = _category_metrics(after, group_counts, after_latency, categories)
        report["configurations"][configuration.name] = {
            "parameters": asdict(configuration),
            "before_rerank": before_metrics,
            "after_rerank": after_metrics,
            "rerank_delta": _metric_deltas(before_metrics, after_metrics),
            "rerank_regression_question_ids": regression_ids,
            "by_category": {
                category: {
                    "before_rerank": before_categories[category],
                    "after_rerank": after_categories[category],
                    "rerank_delta": _metric_deltas(
                        before_categories[category], after_categories[category]
                    ),
                }
                for category in before_categories
            },
            "details": details,
        }
    ranked = sorted(
        report["configurations"].items(),
        key=lambda item: (
            item[1]["after_rerank"]["hit_at_10"],
            item[1]["after_rerank"]["evidence_recall_at_10"],
            item[1]["after_rerank"]["ndcg_at_10"],
            -item[1]["after_rerank"]["median_latency_seconds"],
        ),
        reverse=True,
    )
    report["recommended_configuration"] = ranked[0][0]
    report["strategies"] = {
        name: report["configurations"][name]
        for name in ("fixed-v1", "semantic-v1", "ensemble", "auto")
    }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output = settings.reports_dir / "retrieval_evaluation.json"
    output.write_text(json.dumps(sanitize_for_export(report), indent=2), encoding="utf-8")
    return report
