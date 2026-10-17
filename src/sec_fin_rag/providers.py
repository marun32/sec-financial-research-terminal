from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from openai import OpenAI

from .config import Settings


class NebiusClient:
    def __init__(self, settings: Settings):
        if not settings.nebius_api_key:
            raise ValueError("NEBIUS_API_KEY is not configured")
        self.settings = settings
        self.client = OpenAI(api_key=settings.nebius_api_key, base_url=settings.nebius_base_url)

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
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts[start : start + batch_size],
            )
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda item: item.index)
            )
        return vectors

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
        system = (
            "You are a financial research assistant. Use only the supplied evidence. "
            "Cite claims inline as [Source N]. Never give personalized investment advice. "
            "If the evidence is insufficient, say so explicitly. Preserve reporting periods "
            "and units."
        )
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nEvidence:\n"
                        f"{json.dumps(sources, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content or "Evidence was insufficient to answer."


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
