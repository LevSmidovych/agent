from __future__ import annotations

from pathlib import Path

import yaml

from core.settings import Settings, SettingsManager


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    mgr = SettingsManager(tmp_path / "nonexistent.yaml")
    assert mgr.data.ui_language == "uk"
    assert mgr.data.theme == "light"


def test_load_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        yaml.safe_dump({"ui_language": "en", "theme": "dark"}),
        encoding="utf-8",
    )
    mgr = SettingsManager(path)
    assert mgr.data.ui_language == "en"
    assert mgr.data.theme == "dark"


def test_update_and_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    mgr = SettingsManager(path)
    mgr.update(theme="dark", current_model="llama3.1:8b")
    mgr.save()

    reloaded = SettingsManager(path)
    assert reloaded.data.theme == "dark"
    assert reloaded.data.current_model == "llama3.1:8b"


def test_extra_keys_ignored(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        yaml.safe_dump({"unknown_key": "x", "ui_language": "en"}),
        encoding="utf-8",
    )
    mgr = SettingsManager(path)
    assert mgr.data.ui_language == "en"
