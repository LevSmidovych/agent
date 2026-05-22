from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from memory.classifier import MemoryClassifier, _parse_decision
from memory.long_term import LongTermMemory


class StubEmbedder:
    def __call__(self, input):  # noqa: A002
        return [[0.0, 0.1, 0.2] for _ in input]

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return [0.0, 0.1, 0.2]
        return [[0.0, 0.1, 0.2] for _ in input]

    def embed_documents(self, input):  # noqa: A002
        return [[0.0, 0.1, 0.2] for _ in input]

    def name(self) -> str:
        return "stub"


@pytest.fixture
def long_term(tmp_path: Path) -> LongTermMemory:
    return LongTermMemory(chroma_path=tmp_path / "chroma", embedder=StubEmbedder())


def _llm_with_response(content: str) -> MagicMock:
    llm = MagicMock()
    llm.chat_json = MagicMock(return_value=content)
    return llm


def test_parse_decision_valid_json() -> None:
    assert _parse_decision('{"save": true, "fact": "x"}') == {"save": True, "fact": "x"}


def test_parse_decision_embedded_json() -> None:
    content = 'here you go: {"save": false} thanks'
    assert _parse_decision(content) == {"save": False}


def test_parse_decision_garbage_returns_none() -> None:
    assert _parse_decision("not json at all") is None
    assert _parse_decision("") is None


def test_classifier_saves_fact_when_model_says_yes(long_term: LongTermMemory) -> None:
    llm = _llm_with_response('{"save": true, "fact": "user is vegan"}')
    classifier = MemoryClassifier(llm, "small", long_term, enabled=True)
    try:
        classifier.submit("i am vegan", "noted!", conversation_id=1, profile="cook")
        classifier.flush(timeout=3.0)
    finally:
        classifier.close(timeout=1.0)
    g, _ = long_term.count(profile="cook")
    assert g == 1


def test_classifier_skips_when_model_says_no(long_term: LongTermMemory) -> None:
    llm = _llm_with_response('{"save": false}')
    classifier = MemoryClassifier(llm, "small", long_term, enabled=True)
    try:
        classifier.submit("whats the time", "i cannot tell", conversation_id=1, profile="g")
        classifier.flush(timeout=3.0)
    finally:
        classifier.close(timeout=1.0)
    g, _ = long_term.count(profile="g")
    assert g == 0


def test_classifier_disabled_does_not_save(long_term: LongTermMemory) -> None:
    llm = _llm_with_response('{"save": true, "fact": "x"}')
    classifier = MemoryClassifier(llm, "small", long_term, enabled=False)
    try:
        classifier.submit("hello", "hi", conversation_id=1, profile="g")
        classifier.flush(timeout=1.0)
    finally:
        classifier.close(timeout=1.0)
    g, _ = long_term.count(profile="g")
    assert g == 0
    llm.chat_json.assert_not_called()


def test_classifier_empty_messages_not_submitted(long_term: LongTermMemory) -> None:
    llm = _llm_with_response('{"save": true, "fact": "x"}')
    classifier = MemoryClassifier(llm, "small", long_term, enabled=True)
    try:
        classifier.submit("", "something", conversation_id=1, profile="g")
        classifier.submit("something", "", conversation_id=1, profile="g")
        classifier.flush(timeout=1.0)
    finally:
        classifier.close(timeout=1.0)
    llm.chat_json.assert_not_called()


def test_classifier_survives_llm_failure(long_term: LongTermMemory) -> None:
    llm = MagicMock()
    llm.chat_json = MagicMock(side_effect=RuntimeError("network down"))
    classifier = MemoryClassifier(llm, "small", long_term, enabled=True)
    try:
        classifier.submit("u", "a", conversation_id=1, profile="g")
        classifier.flush(timeout=2.0)
    finally:
        classifier.close(timeout=1.0)
    g, _ = long_term.count(profile="g")
    assert g == 0
