from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

from .config import Settings
from .models import RetrievedChunk
from .providers import LocalReranker, NebiusClient
from .retrieval import PineconeStore, rerank_chunks
from .sparse import SparseBM25Encoder


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    median_latency_seconds: float


def _is_relevant(chunk: RetrievedChunk, item: dict[str, Any]) -> bool:
    if item.get("unanswerable"):
        return False
    ticker_ok = (
        not item.get("expected_tickers") or chunk.metadata.get("ticker") in item["expected_tickers"]
    )
    form_ok = (
        not item.get("expected_forms")
        or chunk.metadata.get("form", "").removesuffix("/A") in item["expected_forms"]
    )
    section = str(chunk.metadata.get("section", "")).lower()
    section_ok = not item.get("expected_sections") or any(
        expected.lower() in section for expected in item["expected_sections"]
    )
    return ticker_ok and form_ok and section_ok


def _metrics(rankings: list[list[bool]], latencies: list[float]) -> RetrievalMetrics:
    def recall_at(values: list[bool], k: int) -> float:
        return float(any(values[:k]))

    def reciprocal_rank(values: list[bool], k: int) -> float:
        return next((1 / (index + 1) for index, value in enumerate(values[:k]) if value), 0.0)

    def ndcg(values: list[bool], k: int) -> float:
        dcg = sum(
            (1 if value else 0) / math.log2(index + 2) for index, value in enumerate(values[:k])
        )
        return dcg  # one expected evidence group per evaluation item, so ideal DCG is 1

    return RetrievalMetrics(
        recall_at_5=statistics.fmean(recall_at(row, 5) for row in rankings),
        recall_at_10=statistics.fmean(recall_at(row, 10) for row in rankings),
        mrr_at_10=statistics.fmean(reciprocal_rank(row, 10) for row in rankings),
        ndcg_at_10=statistics.fmean(ndcg(row, 10) for row in rankings),
        median_latency_seconds=statistics.median(latencies),
    )


def evaluate(settings: Settings) -> dict[str, Any]:
    questions_path = settings.project_root / "data" / "evaluation" / "gold_questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line]
    nebius = NebiusClient(settings)
    sparse = SparseBM25Encoder.load(settings.app_data_dir / "bm25.json")
    store = PineconeStore(settings, nebius, sparse)
    reranker = LocalReranker()
    report: dict[str, Any] = {"question_count": len(questions), "strategies": {}}
    for namespace in ("fixed-v1", "semantic-v1"):
        before, after, before_latency, after_latency = [], [], [], []
        details = []
        for item in questions:
            started = time.perf_counter()
            retrieved = store.query(item["question"], namespace=namespace, top_k=30)
            before_latency.append(time.perf_counter() - started)
            started = time.perf_counter()
            reranked = rerank_chunks(item["question"], retrieved, reranker, top_n=10)
            after_latency.append(before_latency[-1] + time.perf_counter() - started)
            before_flags = [_is_relevant(chunk, item) for chunk in retrieved]
            after_flags = [_is_relevant(chunk, item) for chunk in reranked]
            if not item.get("unanswerable"):
                before.append(before_flags)
                after.append(after_flags)
            details.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "before_relevant_ranks": [i + 1 for i, flag in enumerate(before_flags) if flag],
                    "after_relevant_ranks": [i + 1 for i, flag in enumerate(after_flags) if flag],
                }
            )
        report["strategies"][namespace] = {
            "before_rerank": asdict(_metrics(before, before_latency)),
            "after_rerank": asdict(_metrics(after, after_latency)),
            "details": details,
        }
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output = settings.reports_dir / "retrieval_evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    review_path = settings.reports_dir / "answer_review.csv"
    if not review_path.exists():
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "category",
                    "question",
                    "faithful_0_or_1",
                    "citations_correct_0_or_1",
                    "numeric_accuracy_0_or_1_or_na",
                    "refusal_correct_0_or_1_or_na",
                    "review_notes",
                ],
            )
            writer.writeheader()
            for item in questions:
                writer.writerow(
                    {"id": item["id"], "category": item["category"], "question": item["question"]}
                )
    return report
