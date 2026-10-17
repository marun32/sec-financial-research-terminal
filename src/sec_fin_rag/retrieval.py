from __future__ import annotations

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .models import Chunk, RetrievedChunk
from .providers import LocalReranker, NebiusClient
from .sparse import SparseBM25Encoder


def _scale_sparse(vector: dict[str, Any], weight: float) -> dict[str, Any]:
    return {"indices": vector["indices"], "values": [value * weight for value in vector["values"]]}


class PineconeStore:
    def __init__(self, settings: Settings, embeddings: NebiusClient, sparse: SparseBM25Encoder):
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not configured")
        self.settings = settings
        self.embeddings = embeddings
        self.sparse = sparse
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

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

    @property
    def index(self) -> Any:
        return self.pc.Index(self.settings.pinecone_index)

    def upsert(self, chunks: list[Chunk], namespace: str, batch_size: int = 64) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.ensure_index(len(vectors[0]))
        records: list[dict[str, Any]] = []
        for chunk, dense in zip(chunks, vectors, strict=True):
            metadata: dict[str, Any] = dict(chunk.metadata)
            metadata.update({"text": chunk.text, "strategy": chunk.strategy})
            records.append(
                {
                    "id": chunk.chunk_id,
                    "values": dense,
                    "sparse_values": self.sparse.encode_document(chunk.text),
                    "metadata": metadata,
                }
            )
        for start in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[start : start + batch_size], namespace=namespace)

    def query(
        self,
        question: str,
        namespace: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 30,
        alpha: float = 0.65,
    ) -> list[RetrievedChunk]:
        dense = [value * alpha for value in self.embeddings.embed([question])[0]]
        sparse = _scale_sparse(self.sparse.encode_query(question), 1 - alpha)
        response = self.index.query(
            namespace=namespace,
            vector=dense,
            sparse_vector=sparse,
            filter=filters or None,
            top_k=top_k,
            include_metadata=True,
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


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    reranker: LocalReranker,
    candidate_count: int = 20,
    top_n: int = 8,
) -> list[RetrievedChunk]:
    candidates = chunks[:candidate_count]
    ordering = reranker.rerank(question, [chunk.text for chunk in candidates])
    return [
        RetrievedChunk(
            chunk_id=candidates[index].chunk_id,
            text=candidates[index].text,
            score=score,
            metadata=candidates[index].metadata,
        )
        for index, score in ordering[:top_n]
    ]
