from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pinecone import Pinecone, ServerlessSpec
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from .config import Settings
from .models import Chunk, RetrievedChunk
from .providers import LocalReranker, NebiusClient
from .sparse import SparseBM25Encoder, sparse_encoder_for


def _scale_sparse(vector: dict[str, Any], weight: float) -> dict[str, Any]:
    return {"indices": vector["indices"], "values": [value * weight for value in vector["values"]]}


class PineconeStore:
    def __init__(
        self,
        settings: Settings,
        embeddings: NebiusClient,
        sparse: SparseBM25Encoder | Mapping[str, SparseBM25Encoder],
    ):
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not configured")
        self.settings = settings
        self.embeddings = embeddings
        self.sparse = sparse
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

    def sparse_for(self, namespace: str) -> SparseBM25Encoder:
        return sparse_encoder_for(self.sparse, namespace)

    def ensure_index(self, dimension: int) -> None:
        listing = self.pc.list_indexes()
        names = (
            set(listing.names())
            if hasattr(listing, "names")
            else {item["name"] for item in listing}
        )
        if self.settings.pinecone_index not in names:
            self.pc.create_index(
                name=self.settings.pinecone_index,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            return
        description = self.pc.describe_index(self.settings.pinecone_index)
        existing_dimension = int(getattr(description, "dimension", 0))
        existing_metric = str(getattr(description, "metric", ""))
        if existing_dimension != dimension:
            raise RuntimeError(
                f"Pinecone dimension mismatch: index={existing_dimension}, model={dimension}"
            )
        if existing_metric != "dotproduct":
            raise RuntimeError(
                f"Pinecone metric mismatch: expected dotproduct, found {existing_metric}"
            )

    @property
    def index(self) -> Any:
        return self.pc.Index(self.settings.pinecone_index)

    def upsert(
        self,
        chunks: list[Chunk],
        namespace: str,
        batch_size: int = 64,
        *,
        replace_namespace: bool = False,
    ) -> None:
        if not chunks:
            if replace_namespace:
                self.index.delete(delete_all=True, namespace=namespace)
            return
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.ensure_index(len(vectors[0]))
        records: list[dict[str, Any]] = []
        for chunk, dense in zip(chunks, vectors, strict=True):
            metadata: dict[str, Any] = dict(chunk.metadata)
            metadata.update({"text": chunk.text, "strategy": chunk.strategy})
            record = {
                "id": chunk.chunk_id,
                "values": dense,
                "metadata": metadata,
            }
            sparse_values = self.sparse_for(namespace).encode_document(chunk.text)
            # Pinecone rejects an explicitly empty sparse vector. Such chunks
            # remain valid dense-only candidates in the hybrid index.
            if sparse_values["values"]:
                record["sparse_values"] = sparse_values
            records.append(record)
        # Stable IDs only replace chunks that still exist. Clear a rebuilt
        # namespace after embeddings have succeeded so removed/renumbered
        # chunks cannot remain searchable as stale vectors.
        if replace_namespace:
            self.index.delete(delete_all=True, namespace=namespace)
        for start in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[start : start + batch_size], namespace=namespace)

    def _records(
        self, chunks: list[Chunk], vectors: list[list[float]], namespace: str
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for chunk, dense in zip(chunks, vectors, strict=True):
            metadata: dict[str, Any] = dict(chunk.metadata)
            metadata.update({"text": chunk.text, "strategy": chunk.strategy})
            record: dict[str, Any] = {
                "id": chunk.chunk_id,
                "values": dense,
                "metadata": metadata,
            }
            sparse_values = self.sparse_for(namespace).encode_document(chunk.text)
            if sparse_values["values"]:
                record["sparse_values"] = sparse_values
            records.append(record)
        return records

    def _index_fingerprint(self, chunks: list[Chunk], namespace: str, batch_size: int) -> str:
        sparse = self.sparse_for(namespace)
        digest = hashlib.sha256()
        digest.update(
            (
                f"stream-v1|{namespace}|{self.settings.embedding_model}|{batch_size}|"
                f"{sparse.document_count}|{sparse.average_length}"
            ).encode()
        )
        for token, count in sorted(sparse.df.items()):
            digest.update(f"{token}:{count}\n".encode())
        for chunk in chunks:
            digest.update(f"{chunk.chunk_id}\n".encode())
        return digest.hexdigest()

    def namespace_count(self, namespace: str) -> int:
        stats = self.index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", {}) or {}
        value = namespaces.get(namespace)
        if value is None:
            return 0
        if isinstance(value, dict):
            return int(value.get("vector_count", 0))
        return int(getattr(value, "vector_count", 0))

    def index_streaming(
        self,
        chunks: list[Chunk],
        namespace: str,
        checkpoint_path: Path,
        batch_size: int = 256,
        upsert_batch_size: int = 64,
        *,
        show_progress: bool = True,
        verify_attempts: int = 10,
    ) -> None:
        """Replace a namespace using resumable embed/upsert batches."""
        if not chunks:
            raise ValueError(f"Cannot index empty namespace {namespace}")
        fingerprint = self._index_fingerprint(chunks, namespace, batch_size)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {}
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                checkpoint = {}
        same_run = checkpoint.get("fingerprint") == fingerprint
        next_offset = int(checkpoint.get("next_offset", 0)) if same_run else 0
        if next_offset >= len(chunks) and self.namespace_count(namespace) == len(chunks):
            print(f"      {namespace}: already complete and verified ({len(chunks):,} vectors).")
            return
        if next_offset >= len(chunks):
            next_offset = 0

        retrying = Retrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        progress = tqdm(
            total=len(chunks),
            initial=next_offset,
            unit="vector",
            desc=f"Index {namespace}",
            disable=not show_progress,
        )
        try:
            first_pending_offset = next_offset
            for start in range(next_offset, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors = retrying(self.embeddings.embed, [chunk.text for chunk in batch])
                if start == first_pending_offset:
                    self.ensure_index(len(vectors[0]))
                records = self._records(batch, vectors, namespace)
                if start == 0:
                    self.index.delete(delete_all=True, namespace=namespace)
                for record_start in range(0, len(records), upsert_batch_size):
                    retrying(
                        self.index.upsert,
                        vectors=records[record_start : record_start + upsert_batch_size],
                        namespace=namespace,
                    )
                next_offset = start + len(batch)
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "namespace": namespace,
                            "next_offset": next_offset,
                            "total": len(chunks),
                            "batch_size": batch_size,
                            "upsert_batch_size": upsert_batch_size,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(checkpoint_path)
                progress.update(len(batch))
        finally:
            progress.close()

        observed = -1
        for attempt in range(max(1, verify_attempts)):
            observed = self.namespace_count(namespace)
            if observed == len(chunks):
                break
            if attempt + 1 < verify_attempts:
                time.sleep(2)
        if observed != len(chunks):
            raise RuntimeError(
                f"Pinecone count mismatch for {namespace}: local={len(chunks)}, remote={observed}"
            )
        print(f"      {namespace}: verified {observed:,} vectors.", flush=True)

    def query(
        self,
        question: str,
        namespace: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 30,
        alpha: float = 0.65,
    ) -> list[RetrievedChunk]:
        dense = [value * alpha for value in self.embeddings.embed([question])[0]]
        sparse = _scale_sparse(self.sparse_for(namespace).encode_query(question), 1 - alpha)
        query_args: dict[str, Any] = {
            "namespace": namespace,
            "vector": dense,
            "filter": filters or None,
            "top_k": top_k,
            "include_metadata": True,
        }
        if any(sparse["values"]):
            query_args["sparse_vector"] = sparse
        response = self.index.query(
            **query_args,
        )
        return [
            RetrievedChunk(
                chunk_id=match.id,
                text=(match.metadata or {}).get("text", ""),
                score=float(match.score),
                metadata=dict(match.metadata or {}),
            )
            for match in response.matches
        ]

    def query_score_components(
        self,
        question: str,
        namespace: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> dict[str, dict[str, float | int | None]]:
        """Expose dense-only and sparse-only ranks/scores for ranking diagnosis."""
        dense, sparse = (
            self.query(question, namespace, filters, top_k=top_k, alpha=1.0),
            self.query(question, namespace, filters, top_k=top_k, alpha=0.0),
        )
        result: dict[str, dict[str, float | int | None]] = {}
        for label, chunks in (("dense", dense), ("sparse", sparse)):
            for rank, chunk in enumerate(chunks, start=1):
                row = result.setdefault(
                    chunk.chunk_id,
                    {
                        "dense_rank": None,
                        "dense_score": None,
                        "sparse_rank": None,
                        "sparse_score": None,
                    },
                )
                row[f"{label}_rank"] = rank
                row[f"{label}_score"] = chunk.score
        return result


@dataclass(frozen=True)
class RetrievalDecision:
    strategy: Literal["fixed-v1", "semantic-v1", "ensemble"]
    confidence: float
    reason: str


def select_retrieval_strategy(question: str, tickers: list[str]) -> RetrievalDecision:
    lowered = question.lower()
    if len(tickers) > 1:
        return RetrievalDecision(
            "ensemble", 0.95, "multi-company evidence requires issuer coverage"
        )
    numeric_terms = ("table", "how much", "total", "percent", "%", "year over year")
    if any(term in lowered for term in numeric_terms):
        return RetrievalDecision("fixed-v1", 0.8, "numeric or table-oriented wording")
    if any(term in lowered for term in ("risk", "explain", "why", "strategy", "competition")):
        return RetrievalDecision("semantic-v1", 0.8, "narrative filing wording")
    return RetrievalDecision("ensemble", 0.5, "low-confidence deterministic fallback")


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievedChunk]], rank_constant: int = 60
) -> list[RetrievedChunk]:
    """Fuse and text-deduplicate namespace results with reciprocal-rank fusion."""
    fused: dict[str, tuple[RetrievedChunk, float]] = {}
    for chunks in result_sets:
        for rank, chunk in enumerate(chunks, start=1):
            ticker = str(chunk.metadata.get("ticker", ""))
            normalized = " ".join(chunk.text.split())
            key = hashlib.sha256(f"{ticker}|{normalized}".encode()).hexdigest()
            previous = fused.get(key)
            score = (previous[1] if previous else 0.0) + 1 / (rank_constant + rank)
            representative = previous[0] if previous else chunk
            fused[key] = (representative, score)
    return [
        RetrievedChunk(item.chunk_id, item.text, score, item.metadata)
        for item, score in sorted(fused.values(), key=lambda value: value[1], reverse=True)
    ]


def retrieve_candidates(
    store: PineconeStore,
    question: str,
    strategy: str,
    filters: dict[str, Any] | None,
    top_k: int,
    alpha: float,
) -> list[RetrievedChunk]:
    if strategy != "ensemble":
        return store.query(question, strategy, filters, top_k=top_k, alpha=alpha)
    namespaces = ("fixed-v1", "semantic-v1")
    ticker_values = ((filters or {}).get("ticker") or {}).get("$in", [])
    query_scopes: list[tuple[str, dict[str, Any] | None]] = []
    if len(ticker_values) > 1:
        for namespace in namespaces:
            for ticker in ticker_values:
                scoped = dict(filters or {})
                scoped["ticker"] = {"$eq": ticker}
                query_scopes.append((namespace, scoped))
    else:
        query_scopes = [(namespace, filters) for namespace in namespaces]
    with ThreadPoolExecutor(max_workers=len(query_scopes)) as executor:
        result_sets = list(
            executor.map(
                lambda scope: store.query(question, scope[0], scope[1], top_k=top_k, alpha=alpha),
                query_scopes,
            )
        )
    return reciprocal_rank_fusion(result_sets)[:top_k]


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    reranker: LocalReranker,
    candidate_count: int = 20,
    top_n: int = 8,
    preserve_tickers: list[str] | None = None,
) -> list[RetrievedChunk]:
    candidates = chunks[:candidate_count]
    ordering = reranker.rerank(question, [chunk.text for chunk in candidates])
    reranked = [
        RetrievedChunk(
            chunk_id=candidates[index].chunk_id,
            text=candidates[index].text,
            score=score,
            metadata=candidates[index].metadata,
        )
        for index, score in ordering[:top_n]
    ]
    required = list(dict.fromkeys(preserve_tickers or []))
    if len(required) < 2:
        return reranked
    present = {str(chunk.metadata.get("ticker", "")) for chunk in reranked}
    missing = [ticker for ticker in required if ticker not in present]
    if not missing:
        return reranked
    ranked_all = [
        RetrievedChunk(
            chunk_id=candidates[index].chunk_id,
            text=candidates[index].text,
            score=score,
            metadata=candidates[index].metadata,
        )
        for index, score in ordering
    ]
    protected = set(required) & present
    for ticker in missing:
        replacement = next(
            (chunk for chunk in ranked_all if chunk.metadata.get("ticker") == ticker), None
        )
        if replacement is None:
            continue
        replace_at = next(
            (
                index
                for index in range(len(reranked) - 1, -1, -1)
                if str(reranked[index].metadata.get("ticker", "")) not in protected
            ),
            len(reranked) - 1,
        )
        reranked[replace_at] = replacement
        protected.add(ticker)
    return reranked
