from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tools.files import ListDirectoryTool, ReadFileTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def stop_event() -> threading.Event:
    return threading.Event()


def test_read_file_happy_path(workspace: Path, stop_event: threading.Event) -> None:
    (workspace / "hello.txt").write_text("hello world", encoding="utf-8")
    tool = ReadFileTool(workspace=workspace)
    result = tool.run(stop_event, path="hello.txt")
    assert result.ok
    assert result.output == "hello world"


def test_read_file_rejects_parent_traversal(workspace: Path, stop_event: threading.Event) -> None:
    outside = workspace.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = ReadFileTool(workspace=workspace)
    result = tool.run(stop_event, path="../secret.txt")
    assert not result.ok
    assert "outside" in result.error.lower()


def test_read_file_rejects_absolute_path(workspace: Path, stop_event: threading.Event) -> None:
    outside = workspace.parent / "x.txt"
    outside.write_text("x", encoding="utf-8")
    tool = ReadFileTool(workspace=workspace)
    result = tool.run(stop_event, path=str(outside))
    assert not result.ok


def test_read_file_extension_whitelist(workspace: Path, stop_event: threading.Event) -> None:
    (workspace / "bad.exe").write_bytes(b"MZ")
    tool = ReadFileTool(workspace=workspace)
    result = tool.run(stop_event, path="bad.exe")
    assert not result.ok
    assert "extension" in result.error.lower()


def test_read_file_truncates_large(workspace: Path, stop_event: threading.Event) -> None:
    (workspace / "big.txt").write_text("x" * (200 * 1024), encoding="utf-8")
    tool = ReadFileTool(workspace=workspace)
    result = tool.run(stop_event, path="big.txt")
    assert result.ok
    assert result.metadata["truncated"] is True
    assert "truncated" in result.output


def test_read_file_missing(workspace: Path, stop_event: threading.Event) -> None:
    tool = ReadFileTool(workspace=workspace)
    result = tool.run(stop_event, path="nonexistent.txt")
    assert not result.ok
    assert "not found" in result.error.lower()


def test_list_directory_shows_entries(workspace: Path, stop_event: threading.Event) -> None:
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    (workspace / "sub").mkdir()
    tool = ListDirectoryTool(workspace=workspace)
    result = tool.run(stop_event, path="")
    assert result.ok
    assert "F a.txt" in result.output
    assert "D sub" in result.output


def test_list_directory_rejects_escape(workspace: Path, stop_event: threading.Event) -> None:
    tool = ListDirectoryTool(workspace=workspace)
    result = tool.run(stop_event, path="../")
    assert not result.ok


def test_list_empty_directory(workspace: Path, stop_event: threading.Event) -> None:
    tool = ListDirectoryTool(workspace=workspace)
    result = tool.run(stop_event)
    assert result.ok
    assert "empty" in result.output.lower()
