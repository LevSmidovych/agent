from __future__ import annotations

from pathlib import Path

import pytest

from memory.storage import Storage


@pytest.fixture
def storage(tmp_db_path: Path) -> Storage:
    s = Storage(tmp_db_path)
    yield s
    s.close()


def test_create_conversation_returns_id(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "qwen2.5:14b")
    assert conv_id > 0


def test_add_and_get_messages(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "qwen2.5:14b")
    storage.add_message(conv_id, role="user", content="hello")
    storage.add_message(conv_id, role="assistant", content="hi there")

    messages = storage.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "hi there"


def test_tool_message_serialization(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "qwen2.5:14b")
    storage.add_message(
        conv_id,
        role="tool",
        content="result",
        tool_name="notes.search",
        tool_args={"query": "foo"},
        tool_result="found 2 notes",
    )
    messages = storage.get_messages(conv_id)
    assert messages[0].tool_name == "notes.search"
    assert messages[0].tool_args == {"query": "foo"}
    assert messages[0].tool_result == "found 2 notes"


def test_latest_conversation(storage: Storage) -> None:
    assert storage.latest_conversation() is None
    storage.create_conversation("general", "m1")
    second = storage.create_conversation("cook", "m2")
    latest = storage.latest_conversation()
    assert latest is not None
    assert latest.id == second
    assert latest.profile_name == "cook"


def test_end_conversation(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    assert storage.get_conversation(conv_id).ended_at is None
    storage.end_conversation(conv_id)
    assert storage.get_conversation(conv_id).ended_at is not None


def test_messages_ordered_by_id(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    for i in range(5):
        storage.add_message(conv_id, role="user", content=f"msg {i}")
    messages = storage.get_messages(conv_id)
    assert [m.content for m in messages] == [f"msg {i}" for i in range(5)]


def test_foreign_key_violation(storage: Storage) -> None:
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        storage.add_message(9999, role="user", content="orphan")


def test_update_last_tool_result(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    storage.add_message(
        conv_id, role="tool", content="",
        tool_name="notes_search", tool_args={"q": "x"},
    )
    storage.add_message(
        conv_id, role="tool", content="",
        tool_name="notes_search", tool_args={"q": "y"},
    )
    ok = storage.update_last_tool_result(conv_id, "notes_search", "2 results")
    assert ok is True
    msgs = storage.get_messages(conv_id)
    # Only the last notes_search should be updated
    assert msgs[0].tool_result is None
    assert msgs[1].tool_result == "2 results"
    assert msgs[1].content == "2 results"


def test_update_last_tool_result_missing_tool_returns_false(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    storage.add_message(conv_id, role="user", content="hi")
    assert storage.update_last_tool_result(conv_id, "unknown_tool", "x") is False


def test_update_last_tool_result_scoped_per_conversation(storage: Storage) -> None:
    a = storage.create_conversation("general", "m")
    b = storage.create_conversation("general", "m")
    storage.add_message(a, role="tool", content="", tool_name="t")
    storage.add_message(b, role="tool", content="", tool_name="t")
    storage.update_last_tool_result(a, "t", "result-a")
    msgs_a = storage.get_messages(a)
    msgs_b = storage.get_messages(b)
    assert msgs_a[0].tool_result == "result-a"
    assert msgs_b[0].tool_result is None
