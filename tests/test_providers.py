from types import SimpleNamespace
from typing import Any

from sec_fin_rag.config import Settings
from sec_fin_rag.providers import NebiusClient, _has_uncited_substantive_line


class FakeEmbeddings:
    def create(self, *, model: str, input: list[str]) -> Any:
        del model
        data = [
            SimpleNamespace(index=index, embedding=[float(value)])
            for index, value in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def test_nebius_embeddings_preserve_order_across_concurrent_batches() -> None:
    client = NebiusClient(Settings(nebius_api_key="test", embedding_concurrency=3))
    client.client = SimpleNamespace(embeddings=FakeEmbeddings())  # type: ignore[assignment]
    inputs = [str(index) for index in range(70)]

    assert client.embed(inputs) == [[float(index)] for index in range(70)]


def test_citation_coverage_flags_uncited_summary_and_accepts_cited_rows() -> None:
    assert _has_uncited_substantive_line(
        "Revenue increased by 10% [Source 1].\nThe trend was positive."
    )
    assert not _has_uncited_substantive_line(
        "Revenue increased by 10% [Source 1].\nThe trend was positive [Source 1]."
    )
