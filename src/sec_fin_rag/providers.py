from __future__ import annotations

import json
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langsmith import traceable
from openai import OpenAI

from .citations import normalize_source_markers
from .config import Settings


def _safe_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Exclude provider objects, settings, and credentials from trace serialization."""
    return {key: value for key, value in inputs.items() if key not in {"self", "runtime"}}


def _has_uncited_substantive_line(answer: str) -> bool:
    """Conservatively flag prose/list/table lines that omit an inline source marker."""
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        if line.startswith("|") and not re.search(r"\d|\$|%", line):
            continue
        words = re.findall(r"[A-Za-z]{2,}", line)
        if len(words) >= 4 and not re.search(r"\[Source\s+\d+\]", line, re.IGNORECASE):
            return True
    return False


class NebiusClient:
    def __init__(self, settings: Settings):
        if not settings.nebius_api_key:
            raise ValueError("NEBIUS_API_KEY is not configured")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            timeout=settings.nebius_timeout_seconds,
            max_retries=settings.nebius_max_retries,
        )

    def model_ids(self) -> set[str]:
        return {model.id for model in self.client.models.list().data}

    def validate_models(self) -> None:
        available = self.model_ids()
        missing = {
            self.settings.embedding_model,
            self.settings.chat_model,
        } - available
        if missing:
            raise RuntimeError(f"Required Nebius models unavailable: {', '.join(sorted(missing))}")

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]

        def request(batch: list[str]) -> list[list[float]]:
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

        workers = min(max(1, self.settings.embedding_concurrency), len(batches))
        if workers == 1:
            completed = [request(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                completed = list(executor.map(request, batches))
        return [vector for batch_vectors in completed for vector in batch_vectors]

    def _chat(self, prompt_value: Any) -> AIMessage:
        messages: list[BaseMessage] = prompt_value.to_messages()
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=cast(
                Any,
                [
                    {
                        "role": "system" if message.type == "system" else "user",
                        "content": str(message.content),
                    }
                    for message in messages
                ],
            ),
        )
        return AIMessage(content=response.choices[0].message.content or "")

    def plan_question(
        self, question: str, context: dict[str, Any], issuer_names: list[str]
    ) -> dict[str, Any]:
        """Ask the chat model for research intent, not for an answer or data availability."""
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only a JSON object describing the user's financial "
                        "research request. Keys: tickers (list of primary tickers), "
                        "metrics (list of normalized metric "
                        "names), years (list of explicit calendar years; expand ranges), "
                        "last_n_completed_years (integer), period_basis (annual, quarterly, ytd, "
                        "or unspecified), quarter (1 to 4 or null), operation (lookup, trend, yoy, "
                        "narrative, mixed, market, "
                        "or unsupported), needs_filing_text (boolean). For 'last N years' choose "
                        "completed annual periods unless the question specifies quarters. For a "
                        "range such as 2023 to 2026 include every year. Do not infer that data is "
                        "absent; this is planning only. Current conversation context can resolve "
                        "follow-ups, but an explicit new request takes precedence."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "conversation_context": context,
                            "issuers": issuer_names,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return payload if isinstance(payload, dict) else {}

    def answer_chain(self) -> RunnableSerializable[dict[str, str], str]:
        """Return the LCEL chain used for grounded answer generation."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial research assistant. Use only the supplied evidence. "
                    "Write a concise, analyst-ready response that synthesizes the evidence; do "
                    "not copy raw JSON, metadata, or long source passages into the answer. Lead "
                    "with the direct conclusion, then explain the most decision-useful details. "
                    "Use short paragraphs and compact tables only when they improve comparison. "
                    "Every material factual claim must end with one or more inline citations in "
                    "the exact form [Source N]. Use only source numbers present in the evidence. "
                    "Use separate [Source N] markers for two-source calculations. Do not append "
                    "an uncited recap or key-points section. "
                    "Cite every substantive bullet, table row, calculation, and summary sentence. "
                    "Do not relabel a broad metric such as total revenue as a narrower requested "
                    "metric such as automotive revenue. Do not change signs, periods, or units. "
                    "Never treat omission from retrieved excerpts as proof that a filing or issuer "
                    "has no such disclosure. Say only that the retrieved evidence does not "
                    "establish the requested fact, and do not build comparisons from absence. "
                    "For SEC XBRL, start/end and fact_duration describe the reported amount; "
                    "source_report_end is the source filing's report period, which can differ "
                    "from the fact period for a comparative amount. Never call a comparative "
                    "fact the original-period filing. Filing date and fiscal-year labels "
                    "describe the source filing. A 3-month "
                    "amount is a standalone quarter; 6- and 9-month amounts are cumulative. "
                    "Never present a year-to-date amount as a standalone quarter. For a year "
                    "without a specified quarter, prefer an available full-year 10-K fact. "
                    "For a multi-year growth question, calculate annual YoY only from adjacent "
                    "full-year facts. If the latest requested year lacks a full-year fact but "
                    "matching current/prior interim durations are supplied, calculate their "
                    "YoY growth separately and label it interim, citing both periods. "
                    "Never give personalized investment advice. If evidence is insufficient, say "
                    "so explicitly. Preserve reporting periods and units.",
                ),
                ("user", "Question: {question}\n\nEvidence:\n{evidence}"),
            ]
        )
        return (
            prompt
            | RunnableLambda(self._chat).with_config(run_name="nebius_chat")
            | StrOutputParser()
        )

    @traceable(  # type: ignore[untyped-decorator]
        name="generate_cited_answer", run_type="chain", process_inputs=_safe_trace_inputs
    )
    def answer(self, question: str, contexts: Iterable[dict[str, Any]]) -> str:
        sources = [
            {
                "source": index,
                "issuer": item.get("issuer"),
                "form": item.get("form"),
                "filing_date": item.get("filing_date"),
                "section": item.get("section"),
                "text": item.get("text"),
            }
            for index, item in enumerate(contexts, start=1)
        ]
        evidence = json.dumps(sources, ensure_ascii=False)
        chain = self.answer_chain()
        answer = normalize_source_markers(
            chain.invoke({"question": question, "evidence": evidence})
        )
        valid = {str(index) for index in range(1, len(sources) + 1)}
        markers = set(re.findall(r"\[Source\s+(\d+)\]", answer, flags=re.IGNORECASE))
        if sources and (
            not markers or not markers <= valid or _has_uncited_substantive_line(answer)
        ):
            repair_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Rewrite the draft using only the supplied evidence. Every material "
                        "factual claim must be concise and analyst-ready. Synthesize rather than "
                        "copying raw JSON, metadata, or long evidence passages. Every material "
                        "claim must end with [Source N]. Cite every substantive bullet, table row, "
                        "calculation, and summary sentence. Do not infer a requested submetric "
                        "from a broader metric. Omit uncited recap sections. "
                        "Preserve signs, periods, and units exactly. "
                        "Never infer that a filing or issuer lacks a disclosure merely because it "
                        "is absent from retrieved excerpts; describe that as an evidence "
                        "limitation. "
                        "Never invent a source number. Return only the corrected answer.",
                    ),
                    (
                        "user",
                        "Question: {question}\n\nEvidence: {evidence}\n\nDraft: {draft}",
                    ),
                ]
            )
            repair_chain = (
                repair_prompt
                | RunnableLambda(self._chat).with_config(run_name="citation_repair")
                | StrOutputParser()
            )
            answer = normalize_source_markers(
                repair_chain.invoke({"question": question, "evidence": evidence, "draft": answer})
            )
        return answer

    @traceable(  # type: ignore[untyped-decorator]
        name="judge_rag_answer", run_type="llm", process_inputs=_safe_trace_inputs
    )
    def judge_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Score generation quality in one structured LLM-as-judge call."""
        rubric = {
            "faithfulness": "Every material answer claim is supported by the evidence.",
            "answer_relevancy": "The answer directly and sufficiently addresses the question.",
            "citation_correctness": "Each [Source N] supports the claim immediately preceding it.",
            "citation_completeness": "Every material factual claim has an inline [Source N].",
        }
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG evaluator. Judge only against supplied evidence. "
                        "Return JSON with keys faithfulness, answer_relevancy, "
                        "citation_correctness, citation_completeness. Each value must be an object "
                        "with score (0 or 1) and a concise reason. Do not reward unsupported "
                        "claims."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "evidence": evidence,
                            "rubric": rubric,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        result: dict[str, Any] = {}
        for name in rubric:
            value = parsed.get(name, {})
            score = int(value.get("score", 0)) if isinstance(value, dict) else 0
            result[name] = {
                "score": 1 if score == 1 else 0,
                "reason": str(value.get("reason", "")) if isinstance(value, dict) else "",
            }
        return result


class LocalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        scores = self.model.predict([(query, document) for document in documents])
        return sorted(enumerate(map(float, scores)), key=lambda item: item[1], reverse=True)
