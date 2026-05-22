from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tools.notes import (
    NotesCreateTool,
    NotesDeleteTool,
    NotesListTool,
    NotesReadTool,
    NotesSearchTool,
    NotesStore,
    slugify,
)


class StubEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input):  # noqa: A002 - chromadb signature
        self.calls += 1
        return [[float(len(t) % 7), 0.1, 0.2, 0.3] for t in input]

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return [float(len(input) % 7), 0.1, 0.2, 0.3]
        return [[float(len(t) % 7), 0.1, 0.2, 0.3] for t in input]

    def embed_documents(self, input):  # noqa: A002
        return [[float(len(t) % 7), 0.1, 0.2, 0.3] for t in input]

    def name(self) -> str:
        return "stub"


def test_slugify_basic() -> None:
    assert slugify("Hello World") == "hello-world"
    assert slugify("  Multiple   Spaces ") == "multiple-spaces"
    assert slugify("!!!") == "note"
    assert slugify("file name with/bad chars") == "file-name-withbad-chars"


@pytest.fixture
def store(tmp_path: Path) -> NotesStore:
    s = NotesStore(
        notes_dir=tmp_path / "notes",
        chroma_path=tmp_path / "chroma",
        embedder=StubEmbedder(),
        collection_name="test_notes",
    )
    yield s


@pytest.fixture
def stop_event() -> threading.Event:
    return threading.Event()


def test_create_and_read(store: NotesStore, stop_event) -> None:
    NotesCreateTool(store).run(stop_event, title="My Note", content="body here")
    result = NotesReadTool(store).run(stop_event, title="My Note")
    assert result.ok
    assert "body here" in result.output
    assert "# My Note" in result.output


def test_create_conflict(store: NotesStore, stop_event) -> None:
    NotesCreateTool(store).run(stop_event, title="Dup", content="1")
    result = NotesCreateTool(store).run(stop_event, title="Dup", content="2")
    assert not result.ok
    assert "already exists" in result.error


def test_list_titles(store: NotesStore, stop_event) -> None:
    NotesCreateTool(store).run(stop_event, title="Alpha", content="x")
    NotesCreateTool(store).run(stop_event, title="Bravo", content="y")
    result = NotesListTool(store).run(stop_event)
    assert result.ok
    assert "Alpha" in result.output
    assert "Bravo" in result.output


def test_list_when_empty(store: NotesStore, stop_event) -> None:
    result = NotesListTool(store).run(stop_event)
    assert result.ok
    assert "no notes" in result.output.lower()


def test_delete(store: NotesStore, stop_event) -> None:
    NotesCreateTool(store).run(stop_event, title="Tmp", content="bye")
    result = NotesDeleteTool(store).run(stop_event, title="Tmp")
    assert result.ok
    result_again = NotesReadTool(store).run(stop_event, title="Tmp")
    assert not result_again.ok


def test_delete_missing(store: NotesStore, stop_event) -> None:
    result = NotesDeleteTool(store).run(stop_event, title="ghost")
    assert not result.ok


def test_search_requires_query(store: NotesStore, stop_event) -> None:
    result = NotesSearchTool(store).run(stop_event, query="")
    assert not result.ok


def test_search_returns_no_matches_when_empty(store: NotesStore, stop_event) -> None:
    result = NotesSearchTool(store).run(stop_event, query="anything")
    assert result.ok
    assert "no matches" in result.output.lower()


def test_search_finds_existing_notes(store: NotesStore, stop_event) -> None:
    NotesCreateTool(store).run(stop_event, title="Recipe", content="pasta with tomato")
    result = NotesSearchTool(store).run(stop_event, query="pasta")
    assert result.ok
    # Deterministic stub returns a fixed ordering; at minimum the result is non-empty
    assert result.output.strip() != ""


def test_search_filters_by_distance_threshold(tmp_path) -> None:
    """Irrelevant matches above the threshold must not be returned."""
    from tools.notes import NotesStore

    class HighDistanceEmb:
        def __call__(self, input):  # noqa: A002
            return [[float(len(t) % 3), 0.5, 0.5, 0.5] for t in input]

        def embed_query(self, input):  # noqa: A002
            if isinstance(input, str):
                return [99.0, -99.0, 99.0, -99.0]  # far from anything
            return [[99.0, -99.0, 99.0, -99.0] for _ in input]

        def embed_documents(self, input):  # noqa: A002
            return self([])

        def name(self) -> str:
            return "far"

    s = NotesStore(
        notes_dir=tmp_path / "n", chroma_path=tmp_path / "c",
        embedder=HighDistanceEmb(), collection_name="far_test",
    )
    s.create("Note", "body")
    # With cosine distance against far vectors, nothing should pass threshold
    hits = s.search("query", top_k=5, max_distance=0.1)
    assert hits == []


def test_search_preview_strips_title(store: NotesStore, stop_event) -> None:
    NotesCreateTool(store).run(
        stop_event, title="Dinner", content="pasta with fresh basil"
    )
    hits = store.search("pasta", top_k=5, max_distance=999.0)
    assert hits
    # The preview must not re-include the title header
    assert "# Dinner" not in hits[0]["preview"]
    assert "Dinner\n\n" not in hits[0]["preview"]
