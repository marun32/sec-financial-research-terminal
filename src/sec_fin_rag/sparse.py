from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9._%-]*", text.lower())


def _index(token: str) -> int:
    return (
        int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF
    )


class SparseBM25Encoder:
    def __init__(
        self, document_count: int = 0, average_length: float = 0, df: dict[str, int] | None = None
    ):
        self.document_count = document_count
        self.average_length = average_length
        self.df = df or {}

    def fit(self, documents: list[str]) -> None:
        self.document_count = len(documents)
        lengths = []
        counts: Counter[str] = Counter()
        for document in documents:
            tokens = _tokens(document)
            lengths.append(len(tokens))
            counts.update(set(tokens))
        self.average_length = sum(lengths) / max(1, len(lengths))
        self.df = dict(counts)

    def encode_document(
        self, text: str, k1: float = 1.2, b: float = 0.75
    ) -> dict[str, list[float] | list[int]]:
        counts = Counter(_tokens(text))
        length = sum(counts.values())
        values: dict[int, float] = {}
        for token, frequency in counts.items():
            df = self.df.get(token, 0)
            idf = math.log(1 + (self.document_count - df + 0.5) / (df + 0.5))
            norm = frequency + k1 * (1 - b + b * length / max(1, self.average_length))
            values[_index(token)] = idf * frequency * (k1 + 1) / norm
        ordered = sorted(values)
        return {"indices": ordered, "values": [values[index] for index in ordered]}

    def encode_query(self, text: str) -> dict[str, list[float] | list[int]]:
        values: dict[int, float] = {}
        for token in set(_tokens(text)):
            df = self.df.get(token, 0)
            values[_index(token)] = math.log(1 + (self.document_count - df + 0.5) / (df + 0.5))
        ordered = sorted(values)
        return {"indices": ordered, "values": [values[index] for index in ordered]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "document_count": self.document_count,
                    "average_length": self.average_length,
                    "df": self.df,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SparseBM25Encoder:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            document_count=payload["document_count"],
            average_length=payload["average_length"],
            df=payload["df"],
        )
