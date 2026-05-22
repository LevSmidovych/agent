from __future__ import annotations

from unittest.mock import MagicMock

import ollama
import pytest

from core.embeddings import OllamaEmbedder
from core.exceptions import OllamaConnectionError


@pytest.fixture
def embedder() -> OllamaEmbedder:
    e = OllamaEmbedder("stub-model", host="http://mock")
    e._client = MagicMock()  # type: ignore[assignment]
    return e


def test_embed_happy_path(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    result = embedder.embed(["a", "b"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_empty_input_returns_empty(embedder: OllamaEmbedder) -> None:
    assert embedder.embed([]) == []
    embedder._client.embed.assert_not_called()


def test_embed_marks_unavailable_on_404(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.side_effect = ollama.ResponseError(
        "model not found", status_code=404
    )
    with pytest.raises(OllamaConnectionError):
        embedder.embed(["x"])
    assert embedder.is_available is False


def test_embed_after_unavailable_fast_fails(embedder: OllamaEmbedder) -> None:
    embedder._unavailable = True
    with pytest.raises(OllamaConnectionError):
        embedder.embed(["x"])
    embedder._client.embed.assert_not_called()


def test_embed_missing_embeddings_field(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.return_value = {}
    with pytest.raises(OllamaConnectionError):
        embedder.embed(["x"])


def test_embed_wraps_other_response_errors(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.side_effect = ollama.ResponseError(
        "server error", status_code=500
    )
    with pytest.raises(OllamaConnectionError):
        embedder.embed(["x"])
    # Non-404 should not mark the embedder unavailable
    assert embedder.is_available is True


def test_embed_wraps_generic_exceptions(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.side_effect = RuntimeError("network down")
    with pytest.raises(OllamaConnectionError):
        embedder.embed(["x"])


def test_call_interface_matches_embed(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.return_value = {"embeddings": [[1.0, 2.0]]}
    out = embedder(["hello"])
    assert out == [[1.0, 2.0]]


def test_embed_query_for_string_returns_single_vector(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.return_value = {"embeddings": [[0.5, 0.5]]}
    out = embedder.embed_query("hi")
    assert out == [0.5, 0.5]


def test_embed_query_for_list_returns_list(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.return_value = {"embeddings": [[1.0], [2.0]]}
    out = embedder.embed_query(["a", "b"])
    assert out == [[1.0], [2.0]]


def test_embed_documents(embedder: OllamaEmbedder) -> None:
    embedder._client.embed.return_value = {"embeddings": [[0.1]]}
    assert embedder.embed_documents(["hi"]) == [[0.1]]


def test_name_and_model(embedder: OllamaEmbedder) -> None:
    assert embedder.model == "stub-model"
    assert embedder.name() == "ollama::stub-model"
