"""Bounded short-term conversational memory."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Deque, Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    ts: float = field(default_factory=time)
    meta: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """FIFO buffer of recent messages, sliced per session id."""

    def __init__(self, max_messages: int = 50) -> None:
        self.max_messages = max_messages
        self._sessions: Dict[str, Deque[Message]] = {}

    def _bucket(self, session_id: str) -> Deque[Message]:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_messages)
        return self._sessions[session_id]

    def add(self, session_id: str, role: str, content: str, **meta: str) -> Message:
        msg = Message(role=role, content=content, meta=meta)
        self._bucket(session_id).append(msg)
        return msg

    def history(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        bucket = list(self._bucket(session_id))
        if limit is not None:
            bucket = bucket[-limit:]
        return bucket

    def messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self.history(session_id, limit)]

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)
