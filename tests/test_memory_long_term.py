from __future__ import annotations

from pathlib import Path

import pytest

from memory.long_term import LongTermMemory


class StubEmbedder:
    def __call__(self, input):  # noqa: A002
        return [[float(len(t) % 7), 0.1, 0.2] for t in input]

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return [float(len(input) % 7), 0.1, 0.2]
        return [[float(len(t) % 7), 0.1, 0.2] for t in input]

    def embed_documents(self, input):  # noqa: A002
        return [[float(len(t) % 7), 0.1, 0.2] for t in input]

    def name(self) -> str:
        return "stub"


@pytest.fixture
def memory(tmp_path: Path) -> LongTermMemory:
    return LongTermMemory(chroma_path=tmp_path / "chroma", embedder=StubEmbedder())


def test_add_and_count_global(memory: LongTermMemory) -> None:
    memory.add("user name is Bob", scope="global", source="user")
    g, p = memory.count(profile="cook")
    assert g == 1
    assert p == 0


def test_add_profile_scope(memory: LongTermMemory) -> None:
    memory.add("likes garlic", scope="profile", source="user", profile="cook")
    g, p = memory.count(profile="cook")
    assert g == 0
    assert p == 1


def test_profiles_are_isolated(memory: LongTermMemory) -> None:
    memory.add("a", scope="profile", source="user", profile="cook")
    memory.add("b", scope="profile", source="user", profile="dev")
    _, cook = memory.count(profile="cook")
    _, dev = memory.count(profile="dev")
    assert cook == 1
    assert dev == 1


def test_empty_text_rejected(memory: LongTermMemory) -> None:
    with pytest.raises(ValueError):
        memory.add("", scope="global", source="user")


def test_unknown_scope_rejected(memory: LongTermMemory) -> None:
    with pytest.raises(ValueError):
        memory.add("x", scope="weird", source="user")


def test_unknown_source_rejected(memory: LongTermMemory) -> None:
    with pytest.raises(ValueError):
        memory.add("x", scope="global", source="robot")


def test_profile_scope_requires_profile(memory: LongTermMemory) -> None:
    with pytest.raises(ValueError):
        memory.add("x", scope="profile", source="user", profile=None)


def test_search_returns_results(memory: LongTermMemory) -> None:
    memory.add("allergic to nuts", scope="global", source="user")
    memory.add("likes spicy food", scope="profile", source="user", profile="cook")
    hits = memory.search("nut allergy", profile="cook", top_k=3)
    assert len(hits) >= 1


def test_list_all_filtered_by_scope(memory: LongTermMemory) -> None:
    memory.add("g", scope="global", source="user")
    memory.add("p", scope="profile", source="user", profile="cook")
    all_records = memory.list_all(profile="cook")
    assert len(all_records) == 2
    global_only = memory.list_all(profile="cook", scope_filter="global")
    assert len(global_only) == 1
    assert global_only[0].scope == "global"


def test_delete(memory: LongTermMemory) -> None:
    rid = memory.add("to delete", scope="global", source="user")
    memory.delete(rid, scope="global")
    g, _ = memory.count(profile="cook")
    assert g == 0


def test_safe_profile_name_normalized(memory: LongTermMemory) -> None:
    memory.add("x", scope="profile", source="user", profile="My Profile!")
    memory.add("y", scope="profile", source="user", profile="my_profile")
    # Both should land in the same safe collection
    _, p = memory.count(profile="My Profile!")
    assert p == 2
