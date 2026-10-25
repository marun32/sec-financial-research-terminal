from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path


class SemanticChunkCache:
    """Persistent cache of completed semantic pieces keyed by segment and configuration."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_sections (
                cache_key TEXT PRIMARY KEY,
                pieces_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def get(self, cache_key: str) -> list[str] | None:
        row = self._connection.execute(
            "SELECT pieces_json FROM semantic_sections WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        try:
            payload = json.loads(row[0])
            if not isinstance(payload, list) or not all(
                isinstance(piece, str) for piece in payload
            ):
                raise ValueError("invalid semantic cache entry")
        except (json.JSONDecodeError, TypeError, ValueError):
            with self._connection:
                self._connection.execute(
                    "DELETE FROM semantic_sections WHERE cache_key = ?", (cache_key,)
                )
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def set_many(self, entries: Mapping[str, list[str]]) -> None:
        if not entries:
            return
        rows = [
            (cache_key, json.dumps(pieces, ensure_ascii=False))
            for cache_key, pieces in entries.items()
        ]
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO semantic_sections (cache_key, pieces_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    pieces_json = excluded.pieces_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                rows,
            )
        self.writes += len(rows)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SemanticChunkCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
