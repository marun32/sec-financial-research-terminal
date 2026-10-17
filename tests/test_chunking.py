from sec_fin_rag.chunking import fixed_chunks, semantic_chunks
from sec_fin_rag.models import DocumentBlock


def block(text: str, kind: str = "text") -> DocumentBlock:
    return DocumentBlock(
        block_id="b1",
        text=text,
        block_type=kind,  # type: ignore[arg-type]
        section="Risk Factors",
        metadata={"cik": 1, "accession": "a1", "ticker": "EX"},
    )


def test_fixed_chunks_are_stable_and_overlap() -> None:
    source = block(" ".join(f"word{i}" for i in range(30)))
    first = fixed_chunks([source], max_tokens=10, overlap_tokens=2)
    second = fixed_chunks([source], max_tokens=10, overlap_tokens=2)
    assert len(first) == 4
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert first[0].text.split()[-2:] == first[1].text.split()[:2]


def test_semantic_chunker_preserves_small_table() -> None:
    source = block("| Year | Revenue |\n| --- | --- |\n| 2026 | 100 |", "table")
    chunks = semantic_chunks([source], embed=lambda texts: [[1.0, 0.0] for _ in texts])
    assert len(chunks) == 1
    assert chunks[0].text == source.text
