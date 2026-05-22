from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SlashCommand:
    name: str
    args: str

    def shell_split(self) -> list[str]:
        try:
            return shlex.split(self.args)
        except ValueError:
            return self.args.split()


def parse(text: str) -> Optional[SlashCommand]:
    if not text or not text.startswith("/"):
        return None
    stripped = text[1:].strip()
    if not stripped:
        return None
    parts = stripped.split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return SlashCommand(name=name, args=args.strip())


@dataclass
class CommandResult:
    message: str
    consumed: bool = True  # True if the command was handled and should not go to LLM


class SlashDispatcher:
    """Minimal dispatcher for slash-commands.

    Handlers receive a SlashCommand and return a user-visible message string.
    Returning None means "pass through to the LLM" (i.e. don't consume).
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[SlashCommand], Optional[str]]] = {}

    def register(self, name: str, handler: Callable[[SlashCommand], Optional[str]]) -> None:
        self._handlers[name.lower()] = handler

    def names(self) -> list[str]:
        return sorted(self._handlers.keys())

    def handle(self, text: str) -> Optional[CommandResult]:
        cmd = parse(text)
        if cmd is None:
            return None
        handler = self._handlers.get(cmd.name)
        if handler is None:
            return CommandResult(message=f"unknown command: /{cmd.name}", consumed=True)
        message = handler(cmd)
        if message is None:
            return None
        return CommandResult(message=message, consumed=True)
