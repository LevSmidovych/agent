from __future__ import annotations

from memory.long_term import MemoryRecord
from core.prompt_builder import build_system_prompt


def _record(text: str) -> MemoryRecord:
    return MemoryRecord(
        id="x",
        text=text,
        scope="global",
        source="user",
        timestamp="",
        conversation_id=None,
        profile=None,
    )


def test_no_hits_returns_base() -> None:
    assert build_system_prompt("you are a bot", []) == "you are a bot"


def test_no_hits_empty_base_returns_default() -> None:
    result = build_system_prompt("", [])
    assert "respond in the same language" in result.lower()


def test_hits_without_base_include_default() -> None:
    result = build_system_prompt("", [_record("user is Bob")])
    assert "respond in the same language" in result.lower()
    assert "user is Bob" in result


def test_hits_appended_to_base() -> None:
    result = build_system_prompt("base", [_record("likes tea"), _record("has a dog")])
    assert "base" in result
    assert "likes tea" in result
    assert "has a dog" in result
    assert "About the user" in result


def test_empty_text_records_skipped() -> None:
    result = build_system_prompt("b", [_record(""), _record("keep me")])
    assert "keep me" in result
    # Only one bullet, for non-empty entry
    assert result.count("- ") == 1


# ---- RAG chunks path -----------------------------------------------------


class _FakeChunk:
    def __init__(self, text: str, file: str = "") -> None:
        self.text = text
        self.file = file


def test_rag_chunks_appended_after_base() -> None:
    result = build_system_prompt("You are a bot.", [], [
        _FakeChunk("tomato sauce recipe step 1", file="recipe.md"),
        _FakeChunk("salt to taste", file="recipe.md"),
    ])
    assert "You are a bot." in result
    assert "Relevant context from knowledge base" in result
    assert "tomato sauce" in result
    assert "[recipe.md]" in result


def test_rag_chunks_without_base_include_default() -> None:
    result = build_system_prompt("", [], [_FakeChunk("context text", file="a.md")])
    assert "respond in the same language" in result.lower()
    assert "context text" in result


def test_rag_empty_chunks_do_not_add_block() -> None:
    result = build_system_prompt("base", [], [_FakeChunk("", file="a.md"), _FakeChunk("  ")])
    assert "Relevant context" not in result


def test_memory_and_rag_both_present() -> None:
    result = build_system_prompt(
        "You are a bot.",
        [_record("user speaks Ukrainian")],
        [_FakeChunk("recipe chunk", file="r.md")],
    )
    assert "About the user" in result
    assert "user speaks Ukrainian" in result
    assert "Relevant context" in result
    assert "recipe chunk" in result


def test_rag_chunks_separated_by_hr() -> None:
    result = build_system_prompt("base", [], [
        _FakeChunk("chunk A", file="a.md"),
        _FakeChunk("chunk B", file="a.md"),
    ])
    assert "---" in result
