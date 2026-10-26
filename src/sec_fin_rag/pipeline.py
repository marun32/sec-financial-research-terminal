from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import SemanticChunkCache
from .chunking import fixed_chunks, load_tokenizer, semantic_chunks
from .config import Settings
from .filing_catalog import build_filing_catalog
from .manifest import load_issuers
from .models import Chunk, DocumentBlock, Filing
from .parsing import parse_filing_file, section_distribution_audit
from .providers import NebiusClient
from .retrieval import PineconeStore
from .scoring import derive_company_metrics, score_companies
from .sec_client import SECClient
from .sparse import NAMESPACE_SPARSE_FILES, SparseBM25Encoder, load_sparse_encoders
from .xbrl import build_fundamentals


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def download(settings: Settings) -> None:
    settings.ensure_local_dirs()
    issuers = load_issuers(settings.manifest_path)
    with SECClient(settings.sec_user_agent, settings.raw_dir / "cache") as client:
        for issuer in issuers:
            client.download_issuer(issuer, settings.raw_dir)


def parse(settings: Settings) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    for manifest in settings.raw_dir.glob("filings/*/manifest.json"):
        for payload in json.loads(manifest.read_text(encoding="utf-8")):
            filing = Filing(**payload)
            blocks.extend(parse_filing_file(Path(filing.local_path), filing))
    write_jsonl(settings.processed_dir / "blocks.jsonl", [asdict(block) for block in blocks])
    audit = section_distribution_audit(blocks)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "section_distribution.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    summary = audit["summary"]
    print(
        "Parsed "
        f"{summary['block_count']:,} blocks from {summary['filing_count']:,} filings; "
        f"{summary['risk_factor_block_count']:,} risk-factor blocks; "
        f"{len(summary['missing_10k_risk_label_accessions'])} 10-Ks missing risk labels.",
        flush=True,
    )
    return blocks


def _load_blocks(settings: Settings) -> list[DocumentBlock]:
    return [DocumentBlock(**item) for item in read_jsonl(settings.processed_dir / "blocks.jsonl")]


def build_structured_artifacts(settings: Settings, *, total_stages: int = 3) -> pd.DataFrame:
    """Rebuild XBRL facts, filing catalog, and scorecards without touching chunk indexes."""
    settings.ensure_local_dirs()
    print(f"[1/{total_stages}] Building normalized SEC fundamentals...", flush=True)
    issuers = load_issuers(settings.manifest_path)
    fundamentals = build_fundamentals(
        issuers, settings.raw_dir, settings.app_data_dir / "fundamentals.parquet"
    )
    print(f"[2/{total_stages}] Building authoritative filing catalog...", flush=True)
    build_filing_catalog(settings.raw_dir, fundamentals, settings.app_data_dir / "filings.parquet")
    print(f"[3/{total_stages}] Calculating scorecards...", flush=True)
    score_companies(derive_company_metrics(fundamentals)).to_parquet(
        settings.app_data_dir / "scorecards.parquet", index=False
    )
    return fundamentals


def build_artifacts(settings: Settings) -> tuple[list[Chunk], list[Chunk]]:
    build_structured_artifacts(settings, total_stages=6)
    print("[4/6] Loading the embedding-model tokenizer and building fixed chunks...", flush=True)
    blocks = _load_blocks(settings)
    tokenizer = load_tokenizer(settings.embedding_model)
    nebius = NebiusClient(settings)
    fixed = fixed_chunks(blocks, tokenizer=tokenizer, show_progress=True)
    write_jsonl(
        settings.processed_dir / "chunks_fixed.jsonl", [chunk.to_record() for chunk in fixed]
    )
    print(f"      Saved {len(fixed):,} fixed chunks.", flush=True)
    print("[5/6] Building semantic chunks with Nebius embeddings...", flush=True)
    tokenizer_revision = str(tokenizer.init_kwargs.get("_commit_hash") or "default")
    cache_namespace = f"{settings.embedding_model}@{tokenizer_revision}"
    cache_path = settings.cache_dir / "semantic_sections.sqlite3"
    with SemanticChunkCache(cache_path) as semantic_cache:
        semantic = semantic_chunks(
            blocks,
            embed=nebius.embed,
            tokenizer=tokenizer,
            show_progress=True,
            cache=semantic_cache,
            cache_namespace=cache_namespace,
        )
        print(
            "      Semantic cache: "
            f"{semantic_cache.hits:,} hits, {semantic_cache.misses:,} misses, "
            f"{semantic_cache.writes:,} writes.",
            flush=True,
        )
    write_jsonl(
        settings.processed_dir / "chunks_semantic.jsonl", [chunk.to_record() for chunk in semantic]
    )
    print(f"      Saved {len(semantic):,} semantic chunks.", flush=True)
    print("[6/6] Fitting and saving BM25 statistics...", flush=True)
    for namespace, chunks in (("fixed-v1", fixed), ("semantic-v1", semantic)):
        sparse = SparseBM25Encoder()
        sparse.fit([chunk.text for chunk in chunks])
        sparse.save(settings.app_data_dir / NAMESPACE_SPARSE_FILES[namespace])
    print("      Artifact build complete.", flush=True)
    return fixed, semantic


def _load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**item) for item in read_jsonl(path)]


def index(settings: Settings) -> None:
    nebius = NebiusClient(settings)
    sparse = load_sparse_encoders(settings.app_data_dir)
    store = PineconeStore(settings, nebius, sparse)
    fixed = _load_chunks(settings.processed_dir / "chunks_fixed.jsonl")
    semantic = _load_chunks(settings.processed_dir / "chunks_semantic.jsonl")
    print("[1/2] Streaming fixed-v1 to Pinecone...", flush=True)
    store.index_streaming(
        fixed, "fixed-v1", settings.cache_dir / "index_fixed-v1.json", show_progress=True
    )
    print("[2/2] Streaming semantic-v1 to Pinecone...", flush=True)
    store.index_streaming(
        semantic,
        "semantic-v1",
        settings.cache_dir / "index_semantic-v1.json",
        show_progress=True,
    )
    print("      Pinecone indexing complete.", flush=True)


def load_scorecards(settings: Settings) -> pd.DataFrame:
    path = settings.app_data_dir / "scorecards.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
