from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb

from core.chroma_client import ChromaClientFactory

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".txt", ".md"}
DEFAULT_CHUNK_CHARS = 2000  # ≈ 500 tokens (chars/4)
DEFAULT_OVERLAP_CHARS = 200  # ≈ 50 tokens


@dataclass
class RAGChunk:
    file: str
    chunk_index: int
    text: str
    distance: Optional[float] = None


@dataclass
class FileStatus:
    path: Path
    size: int
    mtime: float
    chunks_indexed: int
    indexed_mtime: Optional[float]

    @property
    def is_indexed(self) -> bool:
        return self.chunks_indexed > 0 and self.indexed_mtime == self.mtime


def chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    text = text.rstrip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - overlap_chars)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


class RAGIndex:
    """Per-profile RAG index backed by a ChromaDB collection."""

    def __init__(
        self,
        kb_path: Path,
        chroma_path: Path,
        embedder,
        collection_name: str,
        chroma_factory: "ChromaClientFactory | None" = None,
    ) -> None:
        self._kb = Path(kb_path)
        self._kb.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        if chroma_factory is None:
            chroma_factory = ChromaClientFactory()
        self._collection = chroma_factory.get_or_create_collection(
            path=Path(chroma_path),
            name=collection_name,
            embedding_function=embedder,
        )

    @property
    def kb_path(self) -> Path:
        return self._kb

    # ---- file listing -------------------------------------------------------

    def list_files(self) -> list[FileStatus]:
        statuses: list[FileStatus] = []
        for path in sorted(self._kb.iterdir() if self._kb.exists() else []):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            chunks_indexed, indexed_mtime = self._file_stats(path)
            stat = path.stat()
            statuses.append(
                FileStatus(
                    path=path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    chunks_indexed=chunks_indexed,
                    indexed_mtime=indexed_mtime,
                )
            )
        return statuses

    def _file_stats(self, path: Path) -> tuple[int, Optional[float]]:
        try:
            existing = self._collection.get(where={"file": str(path.name)})
        except Exception:
            return 0, None
        metas = existing.get("metadatas") or []
        if not metas:
            return 0, None
        indexed_mtime = metas[0].get("mtime") if metas else None
        return len(metas), indexed_mtime

    # ---- mutations ----------------------------------------------------------

    def reindex(self, progress_cb=None) -> tuple[int, int]:
        """Reindex files whose mtime changed since last index. Returns (files, chunks)."""
        if not self._kb.exists():
            return (0, 0)
        if hasattr(self._embedder, "is_available") and not self._embedder.is_available:
            return (0, 0)

        disk_names = {p.name for p in self._kb.iterdir() if p.is_file()}
        indexed = self._all_indexed_files()

        # Remove chunks for files deleted from disk
        for stale in indexed - disk_names:
            self._delete_file(stale)

        files_processed = 0
        chunks_written = 0
        statuses = self.list_files()
        total = len(statuses)
        for i, status in enumerate(statuses):
            if progress_cb:
                progress_cb(i, total, status.path.name)
            if status.is_indexed:
                continue
            try:
                written = self._index_file(status.path)
                chunks_written += written
                files_processed += 1
            except Exception:
                logger.exception("failed to index %s", status.path)
        if progress_cb:
            progress_cb(total, total, None)
        return (files_processed, chunks_written)

    def _all_indexed_files(self) -> set[str]:
        try:
            existing = self._collection.get()
        except Exception:
            return set()
        metas = existing.get("metadatas") or []
        return {m.get("file") for m in metas if m.get("file")}

    def _delete_file(self, filename: str) -> None:
        try:
            res = self._collection.get(where={"file": filename})
            ids = res.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
        except Exception:
            logger.exception("failed to delete chunks for %s", filename)

    def _index_file(self, path: Path) -> int:
        self._delete_file(path.name)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error("cannot read %s: %s", path, exc)
            return 0
        chunks = chunk_text(text)
        if not chunks:
            return 0
        mtime = path.stat().st_mtime
        ids = [f"{path.name}#{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, Any]] = [
            {"file": path.name, "chunk_index": i, "mtime": mtime}
            for i in range(len(chunks))
        ]
        self._collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    # ---- search -------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[RAGChunk]:
        if not query.strip():
            return []
        if hasattr(self._embedder, "is_available") and not self._embedder.is_available:
            return []
        try:
            if self._collection.count() == 0:
                return []
            raw = self._collection.query(query_texts=[query], n_results=top_k)
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
            return []
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        results: list[RAGChunk] = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            results.append(
                RAGChunk(
                    file=(meta or {}).get("file", ""),
                    chunk_index=(meta or {}).get("chunk_index", 0),
                    text=doc,
                    distance=dists[i] if i < len(dists) else None,
                )
            )
        return results
