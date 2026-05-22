from __future__ import annotations

from typing import Iterable

from memory.long_term import MemoryRecord


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful personal assistant.\n"
    "Always respond in the same language the user wrote in (Ukrainian → Ukrainian, "
    "English → English, etc.).\n"
    "Use tools ONLY when the user explicitly asks you to read files, list directories, "
    "create/search/delete notes, or perform similar concrete actions. "
    "Do NOT create notes, files, or other side effects just to answer a knowledge question — "
    "answer directly with your own knowledge unless the user asks you to save something.\n"
    "Keep answers concise and focused on what the user asked."
)


def build_system_prompt(
    base: str,
    memory_hits: Iterable[MemoryRecord] = (),
    rag_chunks: Iterable = (),
) -> str:
    base = (base or "").strip()
    if not base:
        base = DEFAULT_SYSTEM_PROMPT

    parts: list[str] = [base]

    memory_lines = [f"- {r.text}" for r in memory_hits if getattr(r, "text", "")]
    if memory_lines:
        parts.append("About the user (from long-term memory):\n" + "\n".join(memory_lines))

    rag_blocks = []
    for chunk in rag_chunks:
        text = getattr(chunk, "text", "") or ""
        if not text.strip():
            continue
        source = getattr(chunk, "file", "") or ""
        header = f"[{source}]" if source else ""
        rag_blocks.append(f"{header}\n{text}".strip())
    if rag_blocks:
        parts.append("Relevant context from knowledge base:\n\n" + "\n\n---\n\n".join(rag_blocks))

    return "\n\n".join(parts)
