from __future__ import annotations

from pathlib import Path

import pytest

from profiles.rag import RAGIndex, chunk_text


class StubEmbedder:
    def __call__(self, input):  # noqa: A002
        return [[float(len(t) % 5), 0.1, 0.2] for t in input]

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return [float(len(input) % 5), 0.1, 0.2]
        return [[float(len(t) % 5), 0.1, 0.2] for t in input]

    def embed_documents(self, input):  # noqa: A002
        return [[float(len(t) % 5), 0.1, 0.2] for t in input]

    def name(self) -> str:
        return "stub"


@pytest.fixture
def rag(tmp_path: Path, request) -> RAGIndex:
    return RAGIndex(
        kb_path=tmp_path / "kb",
        chroma_path=tmp_path / "chroma",
        embedder=StubEmbedder(),
        collection_name=f"test_rag_{request.node.name}",
    )


def test_chunk_text_short_stays_single() -> None:
    chunks = chunk_text("hello", chunk_chars=100)
    assert chunks == ["hello"]


def test_chunk_text_long_splits_with_overlap() -> None:
    text = "a" * 2500
    chunks = chunk_text(text, chunk_chars=1000, overlap_chars=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n") == []


def test_reindex_skips_when_empty(rag: RAGIndex) -> None:
    files, chunks = rag.reindex()
    assert files == 0
    assert chunks == 0


def test_reindex_picks_up_new_file(rag: RAGIndex) -> None:
    (rag.kb_path / "notes.md").write_text("some knowledge base content", encoding="utf-8")
    files, chunks = rag.reindex()
    assert files == 1
    assert chunks >= 1


def test_reindex_is_idempotent_when_mtime_unchanged(rag: RAGIndex) -> None:
    (rag.kb_path / "a.md").write_text("hello world", encoding="utf-8")
    rag.reindex()
    files2, _ = rag.reindex()
    assert files2 == 0


def test_reindex_updates_on_mtime_change(rag: RAGIndex) -> None:
    import os
    import time

    path = rag.kb_path / "a.md"
    path.write_text("first", encoding="utf-8")
    rag.reindex()
    time.sleep(0.01)
    path.write_text("second version with different content", encoding="utf-8")
    new_mtime = time.time() + 10
    os.utime(path, (new_mtime, new_mtime))
    files, _ = rag.reindex()
    assert files == 1


def test_reindex_removes_deleted_files(rag: RAGIndex) -> None:
    path = rag.kb_path / "a.md"
    path.write_text("gone soon", encoding="utf-8")
    rag.reindex()
    assert rag.list_files()[0].is_indexed
    path.unlink()
    rag.reindex()
    statuses = rag.list_files()
    assert statuses == []


def test_ignores_non_whitelisted_files(rag: RAGIndex) -> None:
    (rag.kb_path / "a.md").write_text("ok", encoding="utf-8")
    (rag.kb_path / "b.pdf").write_bytes(b"%PDF-1.4")
    statuses = rag.list_files()
    assert {s.path.name for s in statuses} == {"a.md"}


def test_search_returns_chunks(rag: RAGIndex) -> None:
    (rag.kb_path / "recipe.md").write_text("tomato sauce with basil and garlic", encoding="utf-8")
    rag.reindex()
    hits = rag.search("tomato", top_k=3)
    assert len(hits) >= 1
    assert "tomato" in hits[0].text


def test_search_empty_query(rag: RAGIndex) -> None:
    assert rag.search("") == []


def test_search_empty_collection(rag: RAGIndex) -> None:
    assert rag.search("anything") == []


def test_list_files_reports_status(rag: RAGIndex) -> None:
    (rag.kb_path / "a.md").write_text("content", encoding="utf-8")
    statuses_before = rag.list_files()
    assert statuses_before[0].chunks_indexed == 0
    rag.reindex()
    statuses_after = rag.list_files()
    assert statuses_after[0].chunks_indexed >= 1
    assert statuses_after[0].is_indexed
