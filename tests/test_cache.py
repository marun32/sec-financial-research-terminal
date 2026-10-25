from pathlib import Path

from sec_fin_rag.cache import SemanticChunkCache


def test_semantic_cache_persists_between_connections(tmp_path: Path) -> None:
    path = tmp_path / "semantic.sqlite3"
    with SemanticChunkCache(path) as cache:
        assert cache.get("section-key") is None
        cache.set_many({"section-key": ["First piece", "Second piece"]})
        assert cache.misses == 1
        assert cache.writes == 1

    with SemanticChunkCache(path) as cache:
        assert cache.get("section-key") == ["First piece", "Second piece"]
        assert cache.hits == 1
