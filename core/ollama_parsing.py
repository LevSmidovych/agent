"""Helpers for normalising ollama-python response payloads.

ollama-python returns chat results in two shapes depending on version and
call form (streaming vs non-streaming):

* As ``dict`` payloads: ``{"message": {"content": ..., "tool_calls": [...]}, ...}``.
* As pydantic-like objects with ``.message.content``, ``.message.tool_calls``.

Every consumer of these chunks (the agent, the LLM client, the benchmark
runner, the rule-based scorer) used to carry its own copy of the
"get content / get tool_calls / coerce arguments" boilerplate.  This module
is the single source of truth.
"""

from __future__ import annotations

import json
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Message-level accessors
# ---------------------------------------------------------------------------


def get_message(payload: Any) -> Any:
    """Return the ``message`` object/dict from a chat response or chunk.

    Returns an empty dict (not ``None``) when the payload is malformed, so
    downstream ``.get(...)`` calls stay safe.
    """
    if isinstance(payload, dict):
        return payload.get("message") or {}
    return getattr(payload, "message", None) or {}


def get_content(payload: Any) -> str:
    """Extract the text content from either a top-level chunk or a message."""
    msg = _maybe_unwrap(payload)
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return getattr(msg, "content", "") or ""


def get_eval_count(payload: Any) -> Optional[int]:
    """Extract ``eval_count`` (output token count from Ollama). Only the
    final chunk of a streaming response carries this.
    """
    if isinstance(payload, dict):
        value = payload.get("eval_count")
    else:
        value = getattr(payload, "eval_count", None)
    return int(value) if isinstance(value, int) else None


def get_raw_tool_calls(payload: Any) -> list[Any]:
    """Return the unparsed ``tool_calls`` list from a chunk or message."""
    msg = _maybe_unwrap(payload)
    if isinstance(msg, dict):
        return msg.get("tool_calls") or []
    return getattr(msg, "tool_calls", None) or []


def _maybe_unwrap(payload: Any) -> Any:
    """If ``payload`` looks like a chunk (has ``message``), unwrap it.
    Otherwise return as-is — caller already has the message.
    """
    if isinstance(payload, dict) and "message" in payload:
        return payload.get("message") or {}
    if not isinstance(payload, dict):
        msg = getattr(payload, "message", None)
        if msg is not None:
            return msg
    return payload


# ---------------------------------------------------------------------------
# Tool call normalisation
# ---------------------------------------------------------------------------


def normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """Convert raw tool-call entries to a uniform ``{name, arguments}`` shape.

    Accepts:
      * OpenAI-shaped ``{"function": {"name", "arguments"}}`` dicts.
      * Flat ``{"name", "arguments"}`` dicts.
      * Objects with ``.function.name`` / ``.function.arguments`` attributes.
      * Objects with ``.name`` / ``.arguments`` attributes.

    Drops entries that have no resolvable name. Arguments are coerced from
    JSON strings if needed.
    """
    out: list[dict[str, Any]] = []
    for tc in raw_calls or []:
        normalised = _normalise_one(tc)
        if normalised is not None:
            out.append(normalised)
    return out


def _normalise_one(entry: Any) -> Optional[dict[str, Any]]:
    if isinstance(entry, dict):
        fn = entry.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            raw_args = fn.get("arguments")
        else:
            name = entry.get("name")
            raw_args = entry.get("arguments")
    else:
        fn = getattr(entry, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            raw_args = getattr(fn, "arguments", None)
        else:
            name = getattr(entry, "name", None)
            raw_args = getattr(entry, "arguments", None)
    if not name:
        return None
    return {"name": name, "arguments": coerce_args(raw_args)}


def coerce_args(raw: Any) -> dict[str, Any]:
    """Return ``raw`` as a dict — parses JSON strings, drops invalid input."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Inline tool-call extraction (Qwen-style fallback)
# ---------------------------------------------------------------------------

# Some Ollama + model combinations (notably qwen2.5 in certain versions) emit
# tool calls as plain text via ``<tool_call>...</tool_call>`` tags instead of
# populating ``message.tool_calls``. We parse both well-formed wrapped tags
# and bare ``{"name": ..., "arguments": ...}`` JSON blocks that show up after
# a stray ``</tool_call>`` closer.

import re as _re

_WRAPPED_RE = _re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    _re.DOTALL,
)
_BARE_RE = _re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}',
    _re.DOTALL,
)


def extract_inline_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls emitted as plain text instead of via the native field.

    Returns the same normalised ``[{name, arguments}]`` shape as
    :func:`normalize_tool_calls`. Falls back to bare JSON detection when no
    ``<tool_call>`` tags are present.
    """
    if not text or "name" not in text:
        return []

    raw_jsons: list[str] = list(_WRAPPED_RE.findall(text))
    if not raw_jsons:
        raw_jsons = list(_BARE_RE.findall(text))

    out: list[dict[str, Any]] = []
    for blob in raw_jsons:
        try:
            obj = json.loads(blob)
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        out.append({
            "name": str(name),
            "arguments": coerce_args(obj.get("arguments")),
        })
    return out


_STRIP_BLOCKS_RE = _re.compile(
    r"<tool_call>.*?</tool_call>|</?tool_call>",
    _re.DOTALL,
)

# Strip bare JSON tool-call objects too — some models emit only the JSON +
# a stray ``</tool_call>`` closer with no opening tag.
_STRIP_BARE_JSON_RE = _re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}',
    _re.DOTALL,
)

# Garbage prefixes the model sometimes prepends to a tool-call block
# (observed in qwen2.5 via Ollama: ``HeaderCode:``, ``iNdEx``, ``Index``…).
_STRIP_NOISY_PREFIX_RE = _re.compile(
    r"^\s*(?:HeaderCode|iNdEx|Index|Tool\s*Call)\s*:?\s*$",
    _re.IGNORECASE | _re.MULTILINE,
)


def strip_inline_tool_blocks(text: str) -> str:
    """Remove ``<tool_call>...</tool_call>`` blocks, stray closer tags, bare
    JSON tool-call objects, and noisy header tokens. Leaves the
    human-readable preamble intact. Used when persisting the assistant turn
    that contained inline tool calls.
    """
    if not text:
        return text
    text = _STRIP_BLOCKS_RE.sub("", text)
    text = _STRIP_BARE_JSON_RE.sub("", text)
    text = _STRIP_NOISY_PREFIX_RE.sub("", text)
    return text.strip()
