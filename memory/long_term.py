from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb

from core.chroma_client import ChromaClientFactory

logger = logging.getLogger(__name__)

GLOBAL_COLLECTION = "memory_global"
PROFILE_COLLECTION_PREFIX = "memory_profile_"

_PROFILE_SAFE_RE = re.compile(r"[^a-z0-9_-]+")


def _safe_profile(profile: str) -> str:
    base = profile.strip().lower().replace(" ", "_")
    return _PROFILE_SAFE_RE.sub("", base) or "default"


def _collection_name(scope: str, profile: Optional[str]) -> str:
    if scope == "global":
        return GLOBAL_COLLECTION
    if scope == "profile":
        if not profile:
            raise ValueError("profile required for scope='profile'")
        return f"{PROFILE_COLLECTION_PREFIX}{_safe_profile(profile)}"
    raise ValueError(f"unknown scope: {scope}")


@dataclass
class MemoryRecord:
    id: str
    text: str
    scope: str  # "global" | "profile"
    source: str  # "auto" | "user"
    timestamp: str
    conversation_id: Optional[int]
    profile: Optional[str]


class LongTermMemory:
    """Persistent memory store backed by ChromaDB.

    Two collections:
      - memory_global — facts that apply across profiles (name, allergies, ...)
      - memory_profile_<name> — facts tied to a specific profile
    """

    def __init__(
        self,
        chroma_path: Path,
        embedder,
        chroma_factory: Optional[ChromaClientFactory] = None,
    ) -> None:
        self._path = Path(chroma_path)
        self._path.mkdir(parents=True, exist_ok=True)
        if chroma_factory is None:
            chroma_factory = ChromaClientFactory()
        self._factory = chroma_factory
        self._embedder = embedder
        self._collections: dict[str, Any] = {}

    def _collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._factory.get_or_create_collection(
                path=self._path,
                name=name,
                embedding_function=self._embedder,
            )
        return self._collections[name]

    # --- mutations --------------------------------------------------------

    def add(
        self,
        text: str,
        scope: str,
        source: str,
        profile: Optional[str] = None,
        conversation_id: Optional[int] = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")
        if scope not in ("global", "profile"):
            raise ValueError(f"unknown scope: {scope}")
        if source not in ("auto", "user"):
            raise ValueError(f"unknown source: {source}")

        record_id = str(uuid.uuid4())
        collection = self._collection(_collection_name(scope, profile))
        collection.add(
            ids=[record_id],
            documents=[text],
            metadatas=[{
                "scope": scope,
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "conversation_id": conversation_id if conversation_id is not None else -1,
                "profile": profile or "",
            }],
        )
        return record_id

    def delete(self, record_id: str, scope: str, profile: Optional[str] = None) -> None:
        collection = self._collection(_collection_name(scope, profile))
        collection.delete(ids=[record_id])

    # --- reads ------------------------------------------------------------

    def count(self, profile: Optional[str] = None) -> tuple[int, int]:
        global_count = self._collection(GLOBAL_COLLECTION).count()
        profile_count = 0
        if profile:
            profile_count = self._collection(_collection_name("profile", profile)).count()
        return global_count, profile_count

    def search(
        self,
        query: str,
        profile: Optional[str] = None,
        top_k: int = 3,
    ) -> list[MemoryRecord]:
        if not query.strip():
            return []
        if hasattr(self._embedder, "is_available") and not self._embedder.is_available:
            return []
        results: list[MemoryRecord] = []
        for scope, collection in self._collections_to_query(profile):
            try:
                if collection.count() == 0:
                    continue
                raw = collection.query(query_texts=[query], n_results=top_k)
            except Exception as exc:
                logger.warning("memory search skipped for %s: %s", scope, exc)
                continue
            results.extend(_rows_from_query(raw, scope))
        results.sort(key=lambda r: (r.source == "auto", r.timestamp), reverse=True)
        return results

    def list_all(
        self,
        profile: Optional[str] = None,
        scope_filter: Optional[str] = None,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for scope, collection in self._collections_to_query(profile):
            if scope_filter and scope != scope_filter:
                continue
            try:
                raw = collection.get()
            except Exception:
                logger.exception("chroma get failed for %s", scope)
                continue
            records.extend(_rows_from_get(raw, scope))
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def _collections_to_query(self, profile: Optional[str]):
        yield "global", self._collection(GLOBAL_COLLECTION)
        if profile:
            yield "profile", self._collection(_collection_name("profile", profile))


def _rows_from_query(raw: dict, scope: str) -> list[MemoryRecord]:
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    records: list[MemoryRecord] = []
    for i, rid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        records.append(_record(rid, doc, meta, scope))
    return records


def _rows_from_get(raw: dict, scope: str) -> list[MemoryRecord]:
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    records: list[MemoryRecord] = []
    for i, rid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        records.append(_record(rid, doc, meta, scope))
    return records


def _record(rid: str, doc: str, meta: dict, scope: str) -> MemoryRecord:
    meta = meta or {}
    return MemoryRecord(
        id=rid,
        text=doc,
        scope=meta.get("scope", scope),
        source=meta.get("source", "user"),
        timestamp=meta.get("timestamp", ""),
        conversation_id=meta.get("conversation_id") if meta.get("conversation_id", -1) != -1 else None,
        profile=meta.get("profile") or None,
    )
