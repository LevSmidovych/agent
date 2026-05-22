"""Integration tests that require a live Ollama instance.

Run with:
    pytest --run-integration

Prerequisites:
    - Ollama running at http://localhost:11434
    - A small model pulled (e.g. `ollama pull qwen2.5:1.5b` or `llama3.2:1b`)
    - `ollama pull nomic-embed-text` for embedding-dependent tests
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


@pytest.fixture
def llm(ollama_host: str):
    from core.llm_client import LLMClient

    return LLMClient(host=ollama_host)


def test_ollama_is_reachable(llm) -> None:
    from core.exceptions import OllamaConnectionError

    try:
        models = llm.list_models()
    except OllamaConnectionError as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")
    assert isinstance(models, list)


def _pick_small_model(llm) -> str:
    models = llm.list_models()
    if not models:
        pytest.skip("no models available; pull one first")
    priorities = ("1b", "1.5b", "3b", "7b", "8b", "14b")
    for size in priorities:
        for m in models:
            if size in m.lower():
                return m
    return models[0]


def test_basic_chat_streaming(llm) -> None:
    model = _pick_small_model(llm)
    tokens = []
    for token in llm.chat_stream(
        model=model,
        messages=[{"role": "user", "content": "Reply with just the word 'ok'."}],
    ):
        tokens.append(token)
        if sum(len(t) for t in tokens) > 20:
            break
    assert tokens, "expected at least one streamed token"


def test_embedding_model_available(llm) -> None:
    from core.embeddings import OllamaEmbedder
    from core.exceptions import OllamaConnectionError

    embedder = OllamaEmbedder("nomic-embed-text", host=llm._host)
    try:
        vectors = embedder.embed(["hello world"])
    except OllamaConnectionError as exc:
        pytest.skip(f"nomic-embed-text not available: {exc}")
    assert len(vectors) == 1
    assert len(vectors[0]) > 100  # real embedding dimensionality


def test_benchmark_runner_smoke(llm) -> None:
    from benchmarks.prompts import SPEED
    from benchmarks.runner import BenchmarkRunner

    model = _pick_small_model(llm)
    runner = BenchmarkRunner(llm)
    metrics = runner.run_one(model, SPEED[0])
    assert metrics.error is None
    assert metrics.ttft_ms is not None
    assert metrics.ttft_ms < 30_000
    assert metrics.output_tokens is not None and metrics.output_tokens > 0
