from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory.migrations import apply_migrations


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    timestamp: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None


@dataclass
class Conversation:
    id: int
    started_at: str
    ended_at: Optional[str]
    profile_name: str
    model_name: str


class Storage:
    def __init__(self, db_path: Path | str) -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.row_factory = sqlite3.Row
        apply_migrations(self._conn)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    # conversations ---------------------------------------------------------

    def create_conversation(self, profile_name: str, model_name: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO conversations(started_at, profile_name, model_name) VALUES (?, ?, ?)",
            (_now(), profile_name, model_name),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def end_conversation(self, conversation_id: int) -> None:
        self._conn.execute(
            "UPDATE conversations SET ended_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )
        self._conn.commit()

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return _row_to_conversation(row) if row else None

    def latest_conversation(self) -> Optional[Conversation]:
        row = self._conn.execute(
            "SELECT * FROM conversations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _row_to_conversation(row) if row else None

    # messages --------------------------------------------------------------

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[str] = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content, tool_name, tool_args, tool_result, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                tool_name,
                json.dumps(tool_args) if tool_args is not None else None,
                tool_result,
                _now(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_messages(self, conversation_id: int) -> list[Message]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [_row_to_message(r) for r in rows]

    def update_last_tool_result(
        self,
        conversation_id: int,
        tool_name: str,
        content: str,
    ) -> bool:
        """Write the tool's output/error back onto the most recent matching
        tool message. Returns True if a row was updated.
        """
        cur = self._conn.execute(
            """
            UPDATE messages
            SET tool_result = ?, content = ?
            WHERE id = (
                SELECT id FROM messages
                WHERE conversation_id = ? AND role = 'tool' AND tool_name = ?
                ORDER BY id DESC LIMIT 1
            )
            """,
            (content, content, conversation_id, tool_name),
        )
        self._conn.commit()
        return cur.rowcount > 0


def _row_to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=row["id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        profile_name=row["profile_name"],
        model_name=row["model_name"],
    )


def _row_to_message(row: sqlite3.Row) -> Message:
    tool_args = json.loads(row["tool_args"]) if row["tool_args"] else None
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        timestamp=row["timestamp"],
        tool_name=row["tool_name"],
        tool_args=tool_args,
        tool_result=row["tool_result"],
    )
