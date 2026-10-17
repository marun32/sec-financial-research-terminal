from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable

import numpy as np

from .models import Chunk, DocumentBlock

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _chunk_id(block: DocumentBlock, strategy: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{block.metadata['cik']}:{block.metadata['accession']}:{strategy}:{ordinal}:{digest}"


def _metadata(block: DocumentBlock) -> dict[str, object]:
    return {**block.metadata, "section": block.section, "block_type": block.block_type}


def _split_table(block: DocumentBlock, max_tokens: int) -> list[str]:
    lines = block.text.splitlines()
    if token_count(block.text) <= max_tokens or len(lines) <= 3:
        return [block.text]
    header = lines[:2]
    parts: list[str] = []
    current = header.copy()
    for row in lines[2:]:
        if token_count("\n".join(current + [row])) > max_tokens and len(current) > 2:
            parts.append("\n".join(current))
            current = header.copy()
        current.append(row)
    if len(current) > 2:
        parts.append("\n".join(current))
    return parts


def fixed_chunks(
    blocks: list[DocumentBlock], max_tokens: int = 512, overlap_tokens: int = 64
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        pieces: list[str]
        if block.block_type == "table":
            pieces = _split_table(block, max_tokens)
        else:
            tokens = block.text.split()
            step = max(1, max_tokens - overlap_tokens)
            pieces = [
                " ".join(tokens[start : start + max_tokens])
                for start in range(0, len(tokens), step)
            ]
        for piece in pieces:
            if piece.strip():
                ordinal = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(block, "fixed", ordinal, piece),
                        text=piece,
                        strategy="fixed",
                        metadata=_metadata(block),
                    )
                )
    return chunks


def _cosine_distance(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left), np.asarray(right)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator == 0 else 1.0 - float(np.dot(a, b) / denominator)


def semantic_chunks(
    blocks: list[DocumentBlock],
    embed: Callable[[list[str]], list[list[float]]],
    percentile: float = 90.0,
    min_tokens: int = 200,
    max_tokens: int = 1000,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        if block.block_type == "table":
            pieces = _split_table(block, max_tokens)
        else:
            sentences = [part.strip() for part in _SENTENCE.split(block.text) if part.strip()]
            if len(sentences) < 2:
                pieces = [block.text]
            else:
                vectors = embed(sentences)
                distances = [
                    _cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
                ]
                threshold = float(np.percentile(distances, percentile)) if distances else math.inf
                pieces = []
                current: list[str] = []
                for index, sentence in enumerate(sentences):
                    projected = token_count(" ".join(current + [sentence]))
                    boundary = index > 0 and distances[index - 1] >= threshold
                    if current and (
                        projected > max_tokens
                        or (boundary and token_count(" ".join(current)) >= min_tokens)
                    ):
                        pieces.append(" ".join(current))
                        current = []
                    current.append(sentence)
                if current:
                    pieces.append(" ".join(current))
        for piece in pieces:
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(block, "semantic", ordinal, piece),
                    text=piece,
                    strategy="semantic",
                    metadata=_metadata(block),
                )
            )
    return chunks
