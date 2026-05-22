from __future__ import annotations

from core.slash_commands import SlashDispatcher, parse


def test_parse_none_when_no_slash() -> None:
    assert parse("hello") is None


def test_parse_empty_slash() -> None:
    assert parse("/") is None
    assert parse("/   ") is None


def test_parse_command_and_args() -> None:
    cmd = parse("/model qwen2.5:14b")
    assert cmd is not None
    assert cmd.name == "model"
    assert cmd.args == "qwen2.5:14b"


def test_parse_case_insensitive_name() -> None:
    cmd = parse("/HELP")
    assert cmd.name == "help"


def test_shell_split_handles_quotes() -> None:
    cmd = parse('/remember "my important fact"')
    assert cmd is not None
    assert cmd.shell_split() == ["my important fact"]


def test_dispatcher_unknown_command_reports() -> None:
    d = SlashDispatcher()
    result = d.handle("/nope")
    assert result is not None
    assert result.consumed
    assert "unknown" in result.message.lower()


def test_dispatcher_handles_registered() -> None:
    d = SlashDispatcher()
    d.register("ping", lambda _cmd: "pong")
    result = d.handle("/ping")
    assert result.consumed
    assert result.message == "pong"


def test_dispatcher_passes_through_non_commands() -> None:
    d = SlashDispatcher()
    assert d.handle("regular message") is None


def test_dispatcher_handler_returning_none_passes_through() -> None:
    d = SlashDispatcher()
    d.register("silent", lambda _cmd: None)
    assert d.handle("/silent") is None
