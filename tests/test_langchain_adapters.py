from typing import Any

from langchain_core.documents import Document

from sec_fin_rag.langchain_adapters import FinancialRetriever, NebiusEmbeddings
from sec_fin_rag.models import RetrievedChunk


class FakeNebius:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def test_nebius_embeddings_implements_langchain_interface() -> None:
    embeddings = NebiusEmbeddings(FakeNebius())  # type: ignore[arg-type]
    assert embeddings.embed_documents(["a", "four"]) == [[1.0], [4.0]]
    assert embeddings.embed_query("abc") == [3.0]


def test_financial_retriever_returns_langchain_documents() -> None:
    observed: dict[str, Any] = {}

    def retrieve(
        question: str, tickers: list[str], filters: dict[str, Any]
    ) -> list[RetrievedChunk]:
        observed.update(question=question, tickers=tickers, filters=filters)
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Revenue was 100.",
                score=0.9,
                metadata={"ticker": "AAPL"},
            )
        ]

    retriever = FinancialRetriever(
        retrieve_fn=retrieve,
        tickers=["AAPL"],
        filters={"form": "10-K"},
    )
    documents = retriever.invoke("What was revenue?")

    assert documents == [
        Document(
            page_content="Revenue was 100.",
            metadata={"ticker": "AAPL", "chunk_id": "chunk-1", "retrieval_score": 0.9},
        )
    ]
    assert observed == {
        "question": "What was revenue?",
        "tickers": ["AAPL"],
        "filters": {"form": "10-K"},
    }
