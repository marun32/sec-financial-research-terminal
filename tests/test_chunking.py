from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sec_fin_rag.cache import SemanticChunkCache
from sec_fin_rag.chunking import fixed_chunks, semantic_chunks, token_count
from sec_fin_rag.models import DocumentBlock


def tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend, unk_token="[UNK]"
    )


def block(text: str, kind: str = "text") -> DocumentBlock:
    return DocumentBlock(
        block_id="b1",
        text=text,
        block_type=kind,  # type: ignore[arg-type]
        section="Risk Factors",
        metadata={
            "snapshot_date": "2026-09-30",
            "cik": 1,
            "accession": "a1",
            "ticker": "EX",
        },
    )


def test_fixed_chunks_are_stable_and_overlap() -> None:
    source = block(" ".join(f"word{i}" for i in range(30)))
    active_tokenizer = tokenizer()
    first = fixed_chunks([source], max_tokens=10, overlap_tokens=2, tokenizer=active_tokenizer)
    second = fixed_chunks([source], max_tokens=10, overlap_tokens=2, tokenizer=active_tokenizer)
    assert len(first) == 4
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert all(token_count(item.text, active_tokenizer) <= 10 for item in first)
    assert first[0].text.split()[-2:] == first[1].text.split()[:2]
    assert first[0].chunk_id.startswith("2026-09-30:1:a1:fixed:000000:")


def test_semantic_chunker_preserves_small_table() -> None:
    source = block("| Year | Revenue |\n| --- | --- |\n| 2026 | 100 |", "table")
    chunks = semantic_chunks(
        [source], embed=lambda texts: [[1.0, 0.0] for _ in texts], tokenizer=tokenizer()
    )
    assert len(chunks) == 1
    assert chunks[0].text == source.text


def test_fixed_chunker_combines_paragraph_blocks_before_splitting() -> None:
    blocks = [block(" ".join(f"p{paragraph}w{i}" for i in range(40))) for paragraph in range(4)]
    active_tokenizer = tokenizer()
    chunks = fixed_chunks(blocks, max_tokens=100, overlap_tokens=10, tokenizer=active_tokenizer)
    assert len(chunks) == 2
    assert all(token_count(chunk.text, active_tokenizer) <= 100 for chunk in chunks)
    assert chunks[0].text.endswith("p1w39")
    assert chunks[1].text.startswith("p2w0")
    assert chunks[0].metadata["source_block_count"] == 4


def test_chunkers_never_combine_different_sections() -> None:
    first = block("alpha " * 40)
    second = DocumentBlock(
        block_id="b2",
        text="beta " * 40,
        block_type="text",
        section="Management Discussion",
        metadata=first.metadata,
    )
    chunks = fixed_chunks([first, second], max_tokens=100, overlap_tokens=10, tokenizer=tokenizer())
    assert len(chunks) == 2
    assert {chunk.metadata["section"] for chunk in chunks} == {
        "Risk Factors",
        "Management Discussion",
    }


def test_semantic_chunker_detects_topic_change_across_paragraphs() -> None:
    blocks = [block(" ".join([f"growth{i}"] * 100) + ".") for i in range(3)] + [
        block(" ".join([f"risk{i}"] * 100) + ".") for i in range(3)
    ]

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if text.startswith("growth") else [0.0, 1.0] for text in texts]

    active_tokenizer = tokenizer()
    chunks = semantic_chunks(
        blocks,
        embed=embed,
        min_tokens=200,
        max_tokens=1000,
        tokenizer=active_tokenizer,
    )
    assert len(chunks) == 2
    assert all(200 <= token_count(chunk.text, active_tokenizer) <= 1000 for chunk in chunks)
    assert "growth" in chunks[0].text and "risk" not in chunks[0].text
    assert "risk" in chunks[1].text and "growth" not in chunks[1].text


def test_semantic_chunker_packs_units_across_sections() -> None:
    source = block("Alpha increased. Beta declined.")
    blocks = [
        DocumentBlock(
            block_id=f"b{index}",
            text=source.text,
            block_type="text",
            section=f"Section {index}",
            metadata=source.metadata,
        )
        for index in range(3)
    ]
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    semantic_chunks(
        blocks,
        embed=embed,
        min_tokens=1,
        max_tokens=100,
        tokenizer=tokenizer(),
        embedding_unit_batch_size=10,
    )
    assert len(calls) == 1
    assert len(calls[0]) == 6


def test_semantic_chunker_reuses_persistent_section_cache(tmp_path: Path) -> None:
    path = tmp_path / "semantic.sqlite3"
    source = block("Alpha increased. Beta declined.")
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    with SemanticChunkCache(path) as cache:
        first = semantic_chunks(
            [source],
            embed=embed,
            min_tokens=1,
            max_tokens=100,
            tokenizer=tokenizer(),
            cache=cache,
            cache_namespace="test-model@revision-1",
        )
        assert cache.misses == 1
        assert cache.writes == 1

    calls.clear()
    with SemanticChunkCache(path) as cache:
        second = semantic_chunks(
            [source],
            embed=embed,
            min_tokens=1,
            max_tokens=100,
            tokenizer=tokenizer(),
            cache=cache,
            cache_namespace="test-model@revision-1",
        )
        assert cache.hits == 1

    assert calls == []
    assert first == second


def test_oversized_tables_repeat_headers() -> None:
    rows = "\n".join(f"| row{i} | {'value ' * 8} |" for i in range(12))
    source = block(f"| Metric | Value |\n| --- | --- |\n{rows}", "table")
    chunks = fixed_chunks([source], table_max_tokens=40, tokenizer=tokenizer())
    assert len(chunks) > 1
    assert all(chunk.text.startswith("| Metric | Value |\n| --- | --- |") for chunk in chunks)
