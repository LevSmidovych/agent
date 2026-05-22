from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Optional

from core.llm_client import LLMClient
from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


CLASSIFIER_SYSTEM_PROMPT = (
    "You classify conversation turns to decide what is worth remembering "
    "long-term about the user. A fact is worth remembering if it describes "
    "the user's identity, preferences, constraints, goals, or ongoing projects "
    "(for example: name, language, allergies, dietary preferences, profession, "
    "tools they use, projects they are working on). Do NOT save ephemeral "
    "questions, greetings, or the assistant's own statements.\n\n"
    "Respond with strict JSON of the form:\n"
    '  {"save": true, "fact": "<concise third-person statement about the user>"}\n'
    'or {"save": false}\n'
    "Return nothing else."
)


@dataclass
class ClassifierTask:
    user_message: str
    assistant_message: str
    conversation_id: Optional[int]
    profile: Optional[str]


class MemoryClassifier:
    """Async worker that inspects each turn and optionally stores a fact."""

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        long_term: LongTermMemory,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._model = model
        self._long_term = long_term
        self._enabled = enabled
        self._queue: "queue.Queue[ClassifierTask]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="memory-classifier"
        )
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def submit(
        self,
        user_message: str,
        assistant_message: str,
        conversation_id: Optional[int] = None,
        profile: Optional[str] = None,
    ) -> None:
        if not self._enabled:
            return
        if not user_message.strip() or not assistant_message.strip():
            return
        self._queue.put(
            ClassifierTask(
                user_message=user_message,
                assistant_message=assistant_message,
                conversation_id=conversation_id,
                profile=profile,
            )
        )

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._classify(task)
            except Exception:
                logger.exception("classifier failed for task")
            finally:
                self._queue.task_done()

    def _classify(self, task: ClassifierTask) -> None:
        if not self._enabled:
            return
        prompt = (
            f"User: {task.user_message}\n"
            f"Assistant: {task.assistant_message}\n\n"
            "Decide whether there is a persistent user fact worth saving."
        )
        try:
            content = self._llm.chat_json(
                model=self._model,
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            logger.exception("classifier LLM call failed")
            return

        decision = _parse_decision(content)
        if not decision or not decision.get("save"):
            return
        fact = (decision.get("fact") or "").strip()
        if not fact:
            return
        try:
            self._long_term.add(
                text=fact,
                scope="global",
                source="auto",
                profile=task.profile,
                conversation_id=task.conversation_id,
            )
            logger.info("classifier saved fact: %s", fact)
        except Exception:
            logger.exception("failed to store auto fact")

    def flush(self, timeout: float = 5.0) -> None:
        deadline = threading.Event()

        def _waiter():
            self._queue.join()
            deadline.set()

        waiter = threading.Thread(target=_waiter, daemon=True)
        waiter.start()
        deadline.wait(timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        self.flush(timeout=timeout)
        self._stop.set()
        self._thread.join(timeout=1.0)


def _parse_decision(content: str) -> Optional[dict]:
    if not content.strip():
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
