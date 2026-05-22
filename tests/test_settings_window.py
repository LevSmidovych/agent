from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_settings_window_loads_current_values(qapp, tmp_path: Path) -> None:
    from core.settings import SettingsManager
    from ui.i18n import I18n
    from ui.settings_window import SettingsWindow

    settings_path = tmp_path / "settings.yaml"
    mgr = SettingsManager(settings_path)
    mgr.update(
        ui_language="en",
        theme="dark",
        current_model="qwen2.5:7b",
        embedding_model="nomic-embed-text",
        short_term_memory_limit=12,
        auto_memory_enabled=False,
        tool_timeout_seconds=45,
    )
    mgr.save()

    i18n = I18n(Path("ui/i18n"), default_language="en")
    dlg = SettingsWindow(mgr, available_models=["qwen2.5:7b"], i18n=i18n)

    assert dlg._lang_box.currentData() == "en"
    assert dlg._theme_box.currentData() == "dark"
    assert dlg._short_limit.value() == 12
    assert dlg._auto_memory.isChecked() is False
    assert dlg._tool_timeout.value() == 45


def test_settings_window_applied_changes_collected(qapp, tmp_path: Path) -> None:
    from core.settings import SettingsManager
    from ui.i18n import I18n
    from ui.settings_window import SettingsWindow

    mgr = SettingsManager(tmp_path / "settings.yaml")
    i18n = I18n(Path("ui/i18n"), default_language="uk")
    dlg = SettingsWindow(mgr, available_models=[], i18n=i18n)

    dlg._lang_box.setCurrentIndex(1)  # en
    dlg._theme_box.setCurrentIndex(1)  # dark
    dlg._short_limit.setValue(10)
    dlg._auto_memory.setChecked(False)
    dlg._tool_timeout.setValue(60)
    dlg._model_box.setEditText("llama3.1:8b")

    # Simulate save without modal Yes/No since embedding didn't change
    dlg._on_save()

    changes = dlg.applied_changes
    assert changes["ui_language"] == "en"
    assert changes["theme"] == "dark"
    assert changes["short_term_memory_limit"] == 10
    assert changes["auto_memory_enabled"] is False
    assert changes["tool_timeout_seconds"] == 60
    assert changes["current_model"] == "llama3.1:8b"
