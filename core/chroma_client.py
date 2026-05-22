from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)


class ChromaClientFactory:
    """Shared ChromaDB client keyed by persistence path.

    Creating multiple ``PersistentClient`` instances pointing at the same
    directory wastes memory and forces ChromaDB to reload metadata each
    time. This factory caches one client per path.
    """

    def __init__(self) -> None:
        self._clients: dict[Path, Any] = {}
        self._lock = threading.Lock()

    def get(self, path: Path) -> Any:
        path = Path(path).resolve()
        with self._lock:
            client = self._clients.get(path)
            if client is None:
                path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(path))
                self._clients[path] = client
            return client

    def get_or_create_collection(
        self,
        path: Path,
        name: str,
        embedding_function,
        metadata: Optional[dict] = None,
    ):
        client = self.get(path)
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata=metadata or {"hnsw:space": "cosine"},
        )
