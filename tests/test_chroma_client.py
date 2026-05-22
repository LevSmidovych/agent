from __future__ import annotations

from pathlib import Path

from core.chroma_client import ChromaClientFactory


class StubEmbedder:
    def __call__(self, input):  # noqa: A002
        return [[0.1, 0.2] for _ in input]

    def embed_query(self, input):  # noqa: A002
        return [0.1, 0.2] if isinstance(input, str) else [[0.1, 0.2] for _ in input]

    def embed_documents(self, input):  # noqa: A002
        return [[0.1, 0.2] for _ in input]

    def name(self) -> str:
        return "stub"


def test_same_path_returns_same_client(tmp_path: Path) -> None:
    factory = ChromaClientFactory()
    a = factory.get(tmp_path / "db")
    b = factory.get(tmp_path / "db")
    assert a is b


def test_different_paths_distinct_clients(tmp_path: Path) -> None:
    factory = ChromaClientFactory()
    a = factory.get(tmp_path / "a")
    b = factory.get(tmp_path / "b")
    assert a is not b


def test_paths_normalized(tmp_path: Path) -> None:
    factory = ChromaClientFactory()
    a = factory.get(tmp_path / "db")
    # Same path in different form should hit cache
    b = factory.get(str(tmp_path) + "/db")
    assert a is b


def test_get_or_create_collection(tmp_path: Path) -> None:
    factory = ChromaClientFactory()
    col = factory.get_or_create_collection(
        path=tmp_path / "db",
        name="test",
        embedding_function=StubEmbedder(),
    )
    assert col is not None
    col.add(ids=["a"], documents=["hi there"])
    assert col.count() == 1


def test_collections_from_factory_share_client(tmp_path: Path) -> None:
    factory = ChromaClientFactory()
    col1 = factory.get_or_create_collection(
        path=tmp_path / "db", name="one", embedding_function=StubEmbedder()
    )
    col2 = factory.get_or_create_collection(
        path=tmp_path / "db", name="two", embedding_function=StubEmbedder()
    )
    col1.add(ids=["x"], documents=["foo"])
    # Retrieving "one" via a fresh call should see the same underlying data
    col1_again = factory.get_or_create_collection(
        path=tmp_path / "db", name="one", embedding_function=StubEmbedder()
    )
    assert col1_again.count() == 1
    assert col2.count() == 0
