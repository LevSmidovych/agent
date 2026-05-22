from __future__ import annotations

from pathlib import Path

import yaml

from profiles.loader import Profile, ProfileManager


def _write(dir_path: Path, name: str, data: dict) -> None:
    (dir_path / name).write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_empty_dir_no_profiles(tmp_path: Path) -> None:
    mgr = ProfileManager(tmp_path)
    assert mgr.keys() == []


def test_loads_yaml_profiles(tmp_path: Path) -> None:
    _write(tmp_path, "general.yaml", {
        "key": "general",
        "name": "General",
        "system_prompt": "be helpful",
        "tools": ["notes"],
    })
    _write(tmp_path, "cook.yaml", {
        "key": "cook",
        "name": "Cook",
        "system_prompt": "help with recipes",
        "tools": ["notes", "files"],
    })
    mgr = ProfileManager(tmp_path)
    assert mgr.keys() == ["cook", "general"]
    cook = mgr.get("cook")
    assert cook is not None
    assert cook.name == "Cook"
    assert cook.tools == ["notes", "files"]


def test_key_defaults_to_stem(tmp_path: Path) -> None:
    _write(tmp_path, "developer.yaml", {"name": "Developer"})
    mgr = ProfileManager(tmp_path)
    assert mgr.get("developer").name == "Developer"
    assert mgr.get("developer").key == "developer"


def test_broken_yaml_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text(": :\n!!invalid", encoding="utf-8")
    _write(tmp_path, "ok.yaml", {"name": "OK"})
    mgr = ProfileManager(tmp_path)
    assert mgr.get("ok") is not None
    assert mgr.get("broken") is None


def test_ensure_returns_fallback(tmp_path: Path) -> None:
    mgr = ProfileManager(tmp_path)
    profile = mgr.ensure("new")
    assert profile.key == "new"
    assert profile.name == "new"


def test_extra_fields_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "x.yaml", {"name": "X", "unknown_field": 42})
    mgr = ProfileManager(tmp_path)
    assert mgr.get("x").name == "X"


def test_reload_picks_up_new_files(tmp_path: Path) -> None:
    mgr = ProfileManager(tmp_path)
    assert mgr.keys() == []
    _write(tmp_path, "late.yaml", {"name": "Late"})
    mgr.reload()
    assert "late" in mgr.keys()
