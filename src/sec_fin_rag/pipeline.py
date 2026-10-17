from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .chunking import fixed_chunks, semantic_chunks
from .config import Settings
from .manifest import load_issuers
from .models import Chunk, DocumentBlock, Filing
from .parsing import parse_filing_file
from .providers import NebiusClient
from .retrieval import PineconeStore
from .scoring import derive_company_metrics, score_companies
from .sec_client import SECClient
from .sparse import SparseBM25Encoder
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
    return blocks


def _load_blocks(settings: Settings) -> list[DocumentBlock]:
    return [DocumentBlock(**item) for item in read_jsonl(settings.processed_dir / "blocks.jsonl")]


def build_artifacts(settings: Settings) -> tuple[list[Chunk], list[Chunk]]:
    settings.ensure_local_dirs()
    issuers = load_issuers(settings.manifest_path)
    fundamentals = build_fundamentals(
        issuers, settings.raw_dir, settings.app_data_dir / "fundamentals.parquet"
    )
    score_companies(derive_company_metrics(fundamentals)).to_parquet(
        settings.app_data_dir / "scorecards.parquet", index=False
    )
    blocks = _load_blocks(settings)
    nebius = NebiusClient(settings)
    fixed = fixed_chunks(blocks)
    semantic = semantic_chunks(blocks, embed=nebius.embed)
    write_jsonl(
        settings.processed_dir / "chunks_fixed.jsonl", [chunk.to_record() for chunk in fixed]
    )
    write_jsonl(
        settings.processed_dir / "chunks_semantic.jsonl", [chunk.to_record() for chunk in semantic]
    )
    sparse = SparseBM25Encoder()
    sparse.fit([chunk.text for chunk in fixed + semantic])
    sparse.save(settings.app_data_dir / "bm25.json")
    return fixed, semantic


def _load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**item) for item in read_jsonl(path)]


def index(settings: Settings) -> None:
    nebius = NebiusClient(settings)
    sparse = SparseBM25Encoder.load(settings.app_data_dir / "bm25.json")
    store = PineconeStore(settings, nebius, sparse)
    fixed = _load_chunks(settings.processed_dir / "chunks_fixed.jsonl")
    semantic = _load_chunks(settings.processed_dir / "chunks_semantic.jsonl")
    store.upsert(fixed, "fixed-v1")
    store.upsert(semantic, "semantic-v1")


def load_scorecards(settings: Settings) -> pd.DataFrame:
    path = settings.app_data_dir / "scorecards.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
