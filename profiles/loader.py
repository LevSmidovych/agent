from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Profile(BaseModel):
    key: str
    name: str
    description: str = ""
    system_prompt: str = ""
    knowledge_base: str = ""
    tools: list[str] = Field(default_factory=list)
    default_language: str = "uk"

    model_config = {"extra": "ignore"}


class ProfileManager:
    def __init__(self, configs_dir: Path) -> None:
        self._dir = Path(configs_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, Profile] = {}
        self._load_all()

    def _load_all(self) -> None:
        self._profiles.clear()
        for path in sorted(self._dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if "key" not in data:
                    data["key"] = path.stem
                if "name" not in data:
                    data["name"] = path.stem
                profile = Profile(**data)
                self._profiles[profile.key] = profile
            except Exception:
                logger.exception("failed to load profile from %s", path)

    def reload(self) -> None:
        self._load_all()

    def keys(self) -> list[str]:
        return sorted(self._profiles.keys())

    def names(self) -> list[str]:
        return [p.name for p in self._profiles.values()]

    def get(self, key: str) -> Optional[Profile]:
        return self._profiles.get(key)

    def ensure(self, key: str) -> Profile:
        profile = self._profiles.get(key)
        if profile is not None:
            return profile
        fallback = Profile(key=key, name=key)
        self._profiles[key] = fallback
        return fallback
