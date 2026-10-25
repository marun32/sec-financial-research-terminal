from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from .models import RetrievedChunk
from .providers import NebiusClient


class NebiusEmbeddings(Embeddings):  # type: ignore[misc]
    """LangChain embedding interface backed by the project's Nebius client."""

    def __init__(self, client: NebiusClient):
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self.client.embed([text])
        if not vectors:
            raise RuntimeError("Nebius returned no query embedding")
        return vectors[0]


def chunk_to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={**chunk.metadata, "chunk_id": chunk.chunk_id, "retrieval_score": chunk.score},
    )


class FinancialRetriever(BaseRetriever):  # type: ignore[misc]
    """LangChain retriever adapter over the existing hybrid retrieval pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retrieve_fn: Callable[[str, list[str], dict[str, Any]], list[RetrievedChunk]] = Field(
        exclude=True
    )
    tickers: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        return [
            chunk_to_document(chunk)
            for chunk in self.retrieve_fn(query, list(self.tickers), dict(self.filters))
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        del run_manager
        return self._get_relevant_documents(
            query, run_manager=CallbackManagerForRetrieverRun.get_noop_manager()
        )

    def with_query_scope(
        self, *, tickers: list[str] | None = None, filters: dict[str, Any] | None = None
    ) -> FinancialRetriever:
        return cast(
            FinancialRetriever,
            self.model_copy(
                update={
                    "tickers": list(tickers or []),
                    "filters": dict(filters or {}),
                }
            ),
        )
