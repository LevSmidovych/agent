from __future__ import annotations

import logging
from typing import Sequence

import ollama

from core.exceptions import OllamaConnectionError

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Embedding function compatible with ChromaDB's interface."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._client = ollama.Client(host=host)
        self._unavailable = False

    @property
    def model(self) -> str:
        return self._model

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002 - chromadb signature
        return self.embed(list(input))

    def embed_query(self, input: str | Sequence[str]):  # noqa: A002
        if isinstance(input, str):
            return self.embed([input])[0]
        return self.embed(list(input))

    def embed_documents(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return self.embed(list(input))

    def name(self) -> str:
        return f"ollama::{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._unavailable:
            raise OllamaConnectionError(
                f"embedding model '{self._model}' is not available; "
                f"pull it first: ollama pull {self._model}"
            )
        try:
            resp = self._client.embed(model=self._model, input=texts)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                if not self._unavailable:
                    logger.warning(
                        "embedding model '%s' is not installed; memory features disabled. "
                        "Run: ollama pull %s",
                        self._model, self._model,
                    )
                self._unavailable = True
                raise OllamaConnectionError(f"embedding model '{self._model}' not found") from exc
            logger.exception("ollama embed failed")
            raise OllamaConnectionError(f"embedding failed: {exc}") from exc
        except Exception as exc:
            logger.exception("ollama embed failed")
            raise OllamaConnectionError(f"embedding failed: {exc}") from exc
        embeddings = resp.get("embeddings") if isinstance(resp, dict) else getattr(resp, "embeddings", None)
        if embeddings is None:
            raise OllamaConnectionError("ollama embed returned no embeddings")
        return [list(e) for e in embeddings]

    @property
    def is_available(self) -> bool:
        return not self._unavailable
