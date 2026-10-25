from types import SimpleNamespace
from typing import Any

import pytest

from sec_fin_rag.config import Settings
from sec_fin_rag.models import Chunk, RetrievedChunk
from sec_fin_rag.providers import LocalReranker
from sec_fin_rag.retrieval import (
    PineconeStore,
    reciprocal_rank_fusion,
    rerank_chunks,
    retrieve_candidates,
    select_retrieval_strategy,
)
from sec_fin_rag.sparse import SparseBM25Encoder


class FakeEmbeddings:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.events.append("embed")
        return [[float(index), 1.0] for index, _text in enumerate(texts)]


class FakeSparse:
    def encode_document(self, text: str) -> dict[str, list[Any]]:
        del text
        return {"indices": [1], "values": [1.0]}

    def encode_query(self, text: str) -> dict[str, list[Any]]:
        del text
        return {"indices": [1], "values": [1.0]}


class EmptySparse(FakeSparse):
    def encode_document(self, text: str) -> dict[str, list[Any]]:
        del text
        return {"indices": [], "values": []}

    def encode_query(self, text: str) -> dict[str, list[Any]]:
        del text
        return {"indices": [], "values": []}


class FakeIndex:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.deletes: list[dict[str, Any]] = []
        self.upserts: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.ids: dict[str, set[str]] = {}

    def delete(self, **kwargs: Any) -> None:
        self.events.append("delete")
        self.deletes.append(kwargs)
        self.ids[kwargs["namespace"]] = set()

    def upsert(self, **kwargs: Any) -> None:
        self.events.append("upsert")
        self.upserts.append(kwargs)
        self.ids.setdefault(kwargs["namespace"], set()).update(
            vector["id"] for vector in kwargs["vectors"]
        )

    def query(self, **kwargs: Any) -> Any:
        self.events.append("query")
        self.queries.append(kwargs)
        return SimpleNamespace(matches=[])

    def describe_index_stats(self) -> Any:
        return SimpleNamespace(
            namespaces={
                namespace: SimpleNamespace(vector_count=len(ids))
                for namespace, ids in self.ids.items()
            }
        )


class FakePinecone:
    def __init__(self, index: FakeIndex) -> None:
        self._index = index

    def list_indexes(self) -> Any:
        return SimpleNamespace(names=lambda: ["test-index"])

    def describe_index(self, name: str) -> Any:
        assert name == "test-index"
        return SimpleNamespace(dimension=2, metric="dotproduct")

    def Index(self, name: str) -> FakeIndex:  # noqa: N802 - mirrors Pinecone SDK
        assert name == "test-index"
        return self._index


class ScopedQueryStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def query(
        self,
        question: str,
        namespace: str,
        filters: dict[str, Any] | None,
        top_k: int,
        alpha: float,
    ) -> list[RetrievedChunk]:
        del question, top_k, alpha
        self.calls.append((namespace, filters))
        ticker = str((filters or {}).get("ticker", {}).get("$eq", ""))
        return [
            RetrievedChunk(
                chunk_id=f"{namespace}:{ticker}",
                text=f"Evidence for {ticker} in {namespace}",
                score=1.0,
                metadata={"ticker": ticker},
            )
        ]


def test_ensemble_retrieval_balances_multi_ticker_query_scopes() -> None:
    store = ScopedQueryStore()
    results = retrieve_candidates(
        store,  # type: ignore[arg-type]
        "Compare Apple and Microsoft",
        "ensemble",
        {"ticker": {"$in": ["AAPL", "MSFT"]}, "form": {"$in": ["10-K"]}},
        top_k=30,
        alpha=0.65,
    )

    assert len(store.calls) == 4
    assert {chunk.metadata["ticker"] for chunk in results} == {"AAPL", "MSFT"}


def test_rebuilt_namespace_is_cleared_after_embedding_before_upsert() -> None:
    events: list[str] = []
    index = FakeIndex(events)
    store = PineconeStore.__new__(PineconeStore)
    store.settings = Settings(pinecone_api_key="test", pinecone_index="test-index")
    store.embeddings = FakeEmbeddings(events)  # type: ignore[assignment]
    store.sparse = FakeSparse()  # type: ignore[assignment]
    store.pc = FakePinecone(index)  # type: ignore[assignment]
    chunks = [
        Chunk(chunk_id="one", text="first", strategy="fixed", metadata={"ticker": "AAPL"}),
        Chunk(chunk_id="two", text="second", strategy="fixed", metadata={"ticker": "AAPL"}),
    ]

    store.upsert(chunks, "fixed-v1", batch_size=1, replace_namespace=True)

    assert events == ["embed", "delete", "upsert", "upsert"]
    assert index.deletes == [{"delete_all": True, "namespace": "fixed-v1"}]
    assert [call["namespace"] for call in index.upserts] == ["fixed-v1", "fixed-v1"]
    assert [call["vectors"][0]["id"] for call in index.upserts] == ["one", "two"]


def test_empty_sparse_vectors_are_omitted_from_upsert() -> None:
    events: list[str] = []
    index = FakeIndex(events)
    store = PineconeStore.__new__(PineconeStore)
    store.settings = Settings(pinecone_api_key="test", pinecone_index="test-index")
    store.embeddings = FakeEmbeddings(events)  # type: ignore[assignment]
    store.sparse = EmptySparse()  # type: ignore[assignment]
    store.pc = FakePinecone(index)  # type: ignore[assignment]
    chunk = Chunk(chunk_id="dense-only", text="...", strategy="fixed", metadata={})

    store.upsert([chunk], "fixed-v1")

    assert "sparse_values" not in index.upserts[0]["vectors"][0]


def test_empty_sparse_vectors_are_omitted_from_query() -> None:
    events: list[str] = []
    index = FakeIndex(events)
    store = PineconeStore.__new__(PineconeStore)
    store.settings = Settings(pinecone_api_key="test", pinecone_index="test-index")
    store.embeddings = FakeEmbeddings(events)  # type: ignore[assignment]
    store.sparse = EmptySparse()  # type: ignore[assignment]
    store.pc = FakePinecone(index)  # type: ignore[assignment]

    assert store.query("...", "fixed-v1") == []

    assert "sparse_vector" not in index.queries[0]


class OrderedReranker(LocalReranker):
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        del query
        return [(index, float(len(documents) - index)) for index in range(len(documents))]


def _retrieved(chunk_id: str, text: str, ticker: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, text, 1.0, {"ticker": ticker})


def test_reranking_preserves_available_cross_company_coverage() -> None:
    candidates = [
        _retrieved("a1", "apple one", "AAPL"),
        _retrieved("a2", "apple two", "AAPL"),
        _retrieved("m1", "microsoft", "MSFT"),
    ]

    result = rerank_chunks(
        "compare",
        candidates,
        OrderedReranker(),
        candidate_count=3,
        top_n=2,
        preserve_tickers=["AAPL", "MSFT"],
    )

    assert {chunk.metadata["ticker"] for chunk in result} == {"AAPL", "MSFT"}


def test_rank_fusion_deduplicates_identical_text() -> None:
    first = [_retrieved("fixed", "same evidence", "AAPL")]
    second = [_retrieved("semantic", "same   evidence", "AAPL")]

    fused = reciprocal_rank_fusion([first, second])

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_auto_strategy_uses_ensemble_for_cross_company_and_low_confidence() -> None:
    assert select_retrieval_strategy("Compare revenue", ["AAPL", "MSFT"]).strategy == "ensemble"
    assert select_retrieval_strategy("Tell me about Apple", ["AAPL"]).strategy == "ensemble"
    assert select_retrieval_strategy("What risk is disclosed?", ["AAPL"]).strategy == "semantic-v1"


class InterruptingEmbeddings(FakeEmbeddings):
    def __init__(self, events: list[str], interrupt_on: int | None = None) -> None:
        super().__init__(events)
        self.calls = 0
        self.interrupt_on = interrupt_on

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.interrupt_on:
            raise KeyboardInterrupt
        return super().embed(texts)


def test_streaming_index_resumes_without_reembedding_successful_batches(tmp_path: Any) -> None:
    events: list[str] = []
    index = FakeIndex(events)
    pc = FakePinecone(index)
    sparse = SparseBM25Encoder()
    sparse.fit(["one", "two", "three", "four", "five"])
    chunks = [
        Chunk(str(index), text, "fixed", {})
        for index, text in enumerate(("one", "two", "three", "four", "five"))
    ]
    checkpoint = tmp_path / "fixed.json"
    store = PineconeStore.__new__(PineconeStore)
    store.settings = Settings(pinecone_api_key="test", pinecone_index="test-index")
    first_embeddings = InterruptingEmbeddings(events, interrupt_on=2)
    store.embeddings = first_embeddings  # type: ignore[assignment]
    store.sparse = sparse
    store.pc = pc  # type: ignore[assignment]

    with pytest.raises(KeyboardInterrupt):
        store.index_streaming(
            chunks, "fixed-v1", checkpoint, batch_size=2, show_progress=False, verify_attempts=1
        )

    resumed_embeddings = InterruptingEmbeddings(events)
    store.embeddings = resumed_embeddings  # type: ignore[assignment]
    store.index_streaming(
        chunks, "fixed-v1", checkpoint, batch_size=2, show_progress=False, verify_attempts=1
    )

    assert first_embeddings.calls == 2
    assert resumed_embeddings.calls == 2
    assert store.namespace_count("fixed-v1") == 5
