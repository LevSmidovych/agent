from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    ui_language: Literal["uk", "en"] = "uk"
    theme: Literal["light", "dark"] = "light"
    current_model: str = "qwen2.5:14b-instruct-q8_0"
    embedding_model: str = "nomic-embed-text"
    classifier_model: str = "qwen2.5:3b-instruct-q4_K_M"
    current_profile: str = "general"
    short_term_memory_limit: int = 20
    auto_memory_enabled: bool = True
    tool_timeout_seconds: int = 30
    workspace_path: str = "data/workspace"
    window_geometry: Optional[dict] = None

    model_config = {"extra": "ignore"}


class SettingsManager:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._settings = self._load()

    def _load(self) -> Settings:
        if not self._path.exists():
            return Settings()
        with self._path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)

    @property
    def data(self) -> Settings:
        return self._settings

    def update(self, **kwargs) -> None:
        self._settings = self._settings.model_copy(update=kwargs)

    def save(self) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._settings.model_dump(),
                f,
                allow_unicode=True,
                sort_keys=False,
            )
