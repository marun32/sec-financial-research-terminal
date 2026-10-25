from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol, cast

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from .models import Chunk, DocumentBlock

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n+")
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-Embedding-8B"
PROSE_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
SEMANTIC_CACHE_VERSION = "semantic-v2"


class SemanticPiecesCache(Protocol):
    hits: int
    misses: int
    writes: int

    def get(self, cache_key: str) -> list[str] | None: ...

    def set_many(self, entries: dict[str, list[str]]) -> None: ...


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str = DEFAULT_TOKENIZER_MODEL) -> Any:
    """Load and cache the tokenizer used by the configured embedding model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def token_count(text: str, tokenizer: Any | None = None) -> int:
    """Count tokens exactly as the embedding model's Hugging Face tokenizer does."""
    active_tokenizer = tokenizer or load_tokenizer()
    return len(active_tokenizer.tokenize(text))


def _splitter(
    tokenizer: Any, max_tokens: int, overlap_tokens: int
) -> RecursiveCharacterTextSplitter:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        separators=PROSE_SEPARATORS,
        keep_separator="end",
        strip_whitespace=True,
    )
    return cast(RecursiveCharacterTextSplitter, splitter)


def _chunk_id(block: DocumentBlock, strategy: str, ordinal: int, text: str) -> str:
    snapshot = str(block.metadata.get("snapshot_date") or "unversioned")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return (
        f"{snapshot}:{block.metadata['cik']}:{block.metadata['accession']}:"
        f"{strategy}:{ordinal:06d}:{digest}"
    )


def _metadata(block: DocumentBlock, ordinal: int, text: str, tokenizer: Any) -> dict[str, object]:
    return {
        **block.metadata,
        "section": block.section,
        "block_type": block.block_type,
        "chunk_number": ordinal,
        "token_count": token_count(text, tokenizer),
    }


def _segment_key(block: DocumentBlock) -> tuple[object, ...]:
    return (
        block.metadata.get("snapshot_date"),
        block.metadata.get("cik"),
        block.metadata.get("accession"),
        block.section,
    )


def _merge_text_blocks(blocks: list[DocumentBlock]) -> DocumentBlock:
    first, last = blocks[0], blocks[-1]
    metadata = dict(first.metadata)
    metadata.update(
        {
            "first_block_id": first.block_id,
            "last_block_id": last.block_id,
            "source_block_count": len(blocks),
        }
    )
    return DocumentBlock(
        block_id=f"{first.block_id}..{last.block_id}",
        text="\n\n".join(block.text for block in blocks),
        block_type="text",
        section=first.section,
        metadata=metadata,
    )


def section_segments(blocks: list[DocumentBlock]) -> list[DocumentBlock]:
    """Combine consecutive paragraphs within a filing section; keep tables isolated."""
    segments: list[DocumentBlock] = []
    pending: list[DocumentBlock] = []

    def flush() -> None:
        if pending:
            segments.append(_merge_text_blocks(pending))
            pending.clear()

    for block in blocks:
        if block.block_type == "table":
            flush()
            segments.append(block)
            continue
        if pending and _segment_key(pending[-1]) != _segment_key(block):
            flush()
        pending.append(block)
    flush()
    return segments


def _split_table(block: DocumentBlock, max_tokens: int, tokenizer: Any) -> list[str]:
    lines = block.text.splitlines()
    if token_count(block.text, tokenizer) <= max_tokens or len(lines) <= 3:
        return [block.text]
    header = lines[:2]
    parts: list[str] = []
    current = header.copy()
    for row in lines[2:]:
        if token_count("\n".join(current + [row]), tokenizer) > max_tokens and len(current) > 2:
            parts.append("\n".join(current))
            current = header.copy()
        current.append(row)
    if len(current) > 2:
        parts.append("\n".join(current))
    return parts


def _next_ordinal(counters: defaultdict[tuple[object, ...], int], block: DocumentBlock) -> int:
    key = (
        block.metadata.get("snapshot_date"),
        block.metadata.get("cik"),
        block.metadata.get("accession"),
    )
    ordinal = counters[key]
    counters[key] += 1
    return ordinal


def fixed_chunks(
    blocks: list[DocumentBlock],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    table_max_tokens: int = 1000,
    tokenizer: Any | None = None,
    show_progress: bool = False,
) -> list[Chunk]:
    active_tokenizer = tokenizer or load_tokenizer()
    prose_splitter = _splitter(active_tokenizer, max_tokens, overlap_tokens)
    chunks: list[Chunk] = []
    counters: defaultdict[tuple[object, ...], int] = defaultdict(int)
    segments = section_segments(blocks)
    iterator = tqdm(segments, desc="Fixed chunking", unit="segment") if show_progress else segments
    for block in iterator:
        pieces = (
            _split_table(block, table_max_tokens, active_tokenizer)
            if block.block_type == "table"
            else prose_splitter.split_text(block.text)
        )
        for piece in pieces:
            if not piece.strip():
                continue
            ordinal = _next_ordinal(counters, block)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(block, "fixed", ordinal, piece),
                    text=piece,
                    strategy="fixed",
                    metadata=_metadata(block, ordinal, piece, active_tokenizer),
                )
            )
    return chunks


def _cosine_distance(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left), np.asarray(right)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator == 0 else 1.0 - float(np.dot(a, b) / denominator)


def _sentence_units(text: str, max_tokens: int, tokenizer: Any) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    units: list[str] = []
    sentence_splitter = _splitter(tokenizer, max_tokens, 0)
    for sentence in sentences or [text]:
        if token_count(sentence, tokenizer) <= max_tokens:
            units.append(sentence)
        else:
            units.extend(sentence_splitter.split_text(sentence))
    return units


def _merge_small_pieces(
    pieces: list[str], min_tokens: int, max_tokens: int, tokenizer: Any
) -> list[str]:
    merged = list(pieces)
    index = 0
    while index < len(merged) and len(merged) > 1:
        if token_count(merged[index], tokenizer) >= min_tokens:
            index += 1
            continue
        if index > 0:
            combined = f"{merged[index - 1]} {merged[index]}"
            if token_count(combined, tokenizer) <= max_tokens:
                merged[index - 1] = combined
                merged.pop(index)
                index = max(0, index - 1)
                continue
        if index + 1 < len(merged):
            combined = f"{merged[index]} {merged[index + 1]}"
            if token_count(combined, tokenizer) <= max_tokens:
                merged[index] = combined
                merged.pop(index + 1)
                continue
        index += 1
    return merged


def _semantic_pieces_from_vectors(
    units: list[str],
    vectors: list[list[float]],
    percentile: float,
    min_tokens: int,
    max_tokens: int,
    tokenizer: Any,
) -> list[str]:
    if len(units) <= 1:
        return units
    if len(vectors) != len(units):
        raise ValueError("embedding provider returned an unexpected vector count")
    distances = [
        _cosine_distance(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)
    ]
    threshold = float(np.percentile(distances, percentile)) if distances else math.inf
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for index, unit in enumerate(units):
        unit_tokens = token_count(unit, tokenizer)
        boundary = index > 0 and distances[index - 1] >= threshold and distances[index - 1] > 1e-9
        if current and (
            current_tokens + unit_tokens > max_tokens or (boundary and current_tokens >= min_tokens)
        ):
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        pieces.append(" ".join(current))
    return _merge_small_pieces(pieces, min_tokens, max_tokens, tokenizer)


def _semantic_cache_key(
    block: DocumentBlock,
    cache_namespace: str,
    percentile: float,
    min_tokens: int,
    max_tokens: int,
) -> str:
    payload = {
        "version": SEMANTIC_CACHE_VERSION,
        "namespace": cache_namespace,
        "percentile": percentile,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "snapshot_date": block.metadata.get("snapshot_date"),
        "cik": block.metadata.get("cik"),
        "accession": block.metadata.get("accession"),
        "section": block.section,
        "text_sha256": hashlib.sha256(block.text.encode("utf-8")).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_chunks(
    blocks: list[DocumentBlock],
    embed: Callable[[list[str]], list[list[float]]],
    percentile: float = 90.0,
    min_tokens: int = 200,
    max_tokens: int = 1000,
    table_max_tokens: int = 1000,
    tokenizer: Any | None = None,
    show_progress: bool = False,
    embedding_unit_batch_size: int = 512,
    cache: SemanticPiecesCache | None = None,
    cache_namespace: str = DEFAULT_TOKENIZER_MODEL,
) -> list[Chunk]:
    if embedding_unit_batch_size < 1:
        raise ValueError("embedding_unit_batch_size must be positive")
    active_tokenizer = tokenizer or load_tokenizer()
    chunks: list[Chunk] = []
    counters: defaultdict[tuple[object, ...], int] = defaultdict(int)
    segments = section_segments(blocks)
    progress = (
        tqdm(total=len(segments), desc="Semantic chunking", unit="segment")
        if show_progress
        else None
    )
    pieces_by_segment: dict[int, list[str]] = {}
    pending: list[tuple[int, list[str], str | None]] = []
    pending_unit_count = 0

    def update_progress(count: int) -> None:
        if progress is None:
            return
        progress.update(count)
        if cache is not None:
            progress.set_postfix(cache_hits=cache.hits, cache_misses=cache.misses)

    def flush_pending() -> None:
        nonlocal pending_unit_count
        if not pending:
            return
        all_units = [unit for _, units, _ in pending for unit in units]
        all_vectors = embed(all_units)
        if len(all_vectors) != len(all_units):
            raise ValueError("embedding provider returned an unexpected vector count")
        cursor = 0
        cache_entries: dict[str, list[str]] = {}
        for segment_index, units, cache_key in pending:
            end = cursor + len(units)
            pieces = _semantic_pieces_from_vectors(
                units,
                all_vectors[cursor:end],
                percentile,
                min_tokens,
                max_tokens,
                active_tokenizer,
            )
            pieces_by_segment[segment_index] = pieces
            if cache_key is not None:
                cache_entries[cache_key] = pieces
            cursor = end
        if cache is not None:
            cache.set_many(cache_entries)
        update_progress(len(pending))
        pending.clear()
        pending_unit_count = 0

    try:
        for segment_index, block in enumerate(segments):
            if block.block_type == "table":
                pieces_by_segment[segment_index] = _split_table(
                    block, table_max_tokens, active_tokenizer
                )
                update_progress(1)
                continue
            cache_key = _semantic_cache_key(
                block,
                cache_namespace,
                percentile,
                min_tokens,
                max_tokens,
            )
            if cache is not None:
                cached = cache.get(cache_key)
                if cached is not None:
                    pieces_by_segment[segment_index] = cached
                    update_progress(1)
                    continue
            units = _sentence_units(block.text, max_tokens, active_tokenizer)
            if len(units) <= 1:
                pieces_by_segment[segment_index] = units
                if cache is not None:
                    cache.set_many({cache_key: units})
                update_progress(1)
                continue
            if pending and pending_unit_count + len(units) > embedding_unit_batch_size:
                flush_pending()
            pending.append((segment_index, units, cache_key if cache is not None else None))
            pending_unit_count += len(units)
            if pending_unit_count >= embedding_unit_batch_size:
                flush_pending()
        flush_pending()
    finally:
        if progress is not None:
            progress.close()

    for segment_index, block in enumerate(segments):
        pieces = pieces_by_segment[segment_index]
        for piece in pieces:
            if not piece.strip():
                continue
            ordinal = _next_ordinal(counters, block)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(block, "semantic", ordinal, piece),
                    text=piece,
                    strategy="semantic",
                    metadata=_metadata(block, ordinal, piece, active_tokenizer),
                )
            )
    return chunks
