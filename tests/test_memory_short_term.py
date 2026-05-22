from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.llm_client import ModelInfo
from memory.short_term import ShortTermMemory
from memory.storage import Storage


@pytest.fixture
def storage(tmp_db_path: Path) -> Storage:
    s = Storage(tmp_db_path)
    yield s
    s.close()


def _fake_llm(context_length: int = 8192, tokens_per_msg: int = 10) -> MagicMock:
    llm = MagicMock()
    llm.model_info = MagicMock(return_value=ModelInfo(name="m", context_length=context_length))
    llm.tokenize = MagicMock(side_effect=lambda model, text: tokens_per_msg)
    return llm


def test_build_empty_conversation(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=20)
    ctx = stm.build(conv_id, model="m")
    assert ctx.messages == []
    assert ctx.truncated == 0


def test_build_includes_system_prompt(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    storage.add_message(conv_id, role="user", content="hi")
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=20)
    ctx = stm.build(conv_id, model="m", system_prompt="you are a bot")
    assert ctx.messages[0] == {"role": "system", "content": "you are a bot"}
    assert ctx.messages[1]["content"] == "hi"


def test_build_skips_tool_messages(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    storage.add_message(conv_id, role="user", content="hi")
    storage.add_message(conv_id, role="tool", content="tool result", tool_name="notes_search")
    storage.add_message(conv_id, role="assistant", content="ok")
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=20)
    ctx = stm.build(conv_id, model="m")
    roles = [m["role"] for m in ctx.messages]
    assert roles == ["user", "assistant"]


def test_message_limit_trims_oldest(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    for i in range(25):
        storage.add_message(conv_id, role="user", content=f"u{i}")
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=5)
    ctx = stm.build(conv_id, model="m")
    assert len(ctx.messages) == 5
    assert ctx.truncated == 20
    assert ctx.messages[-1]["content"] == "u24"


def test_sliding_window_by_token_budget(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    for i in range(10):
        storage.add_message(conv_id, role="user", content=f"u{i}")
    # budget = 4096/2 = 2048, but each msg is 1000 tokens → only 2 fit
    stm = ShortTermMemory(
        storage,
        _fake_llm(context_length=4096, tokens_per_msg=1000),
        message_limit=20,
    )
    ctx = stm.build(conv_id, model="m")
    assert len(ctx.messages) <= 3
    assert ctx.truncated >= 7


def test_drop_last_user_message(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    storage.add_message(conv_id, role="user", content="first")
    storage.add_message(conv_id, role="assistant", content="answer")
    storage.add_message(conv_id, role="user", content="latest")
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=20)
    ctx = stm.build(conv_id, model="m", drop_last=True)
    assert [m["content"] for m in ctx.messages] == ["first", "answer"]


def test_message_limit_setter_clamps_to_min(storage: Storage) -> None:
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=20)
    stm.message_limit = 0
    assert stm.message_limit >= 2


def test_message_limit_setter_updates_behavior(storage: Storage) -> None:
    conv_id = storage.create_conversation("general", "m")
    for i in range(30):
        storage.add_message(conv_id, role="user", content=f"u{i}")
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=20)
    assert len(stm.build(conv_id, model="m").messages) == 20
    stm.message_limit = 5
    assert len(stm.build(conv_id, model="m").messages) == 5


def test_constructor_clamps_message_limit(storage: Storage) -> None:
    stm = ShortTermMemory(storage, _fake_llm(), message_limit=1)
    assert stm.message_limit == 2
