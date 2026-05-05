"""Bounded short-term conversational memory.

Supports two calling styles:
  1. 3-arg session-based:  add(session_id, role, content)  → messages(session_id)
  2. 2-arg ring-buffer:    add(role, content)              → .messages property

Both styles co-exist via a default session id for the ring-buffer API.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Any, Deque, Dict, Iterator, List, Optional

_DEFAULT_SID = "_default"


@dataclass
class Message:
    role: str
    content: str
    ts: float = field(default_factory=time)
    meta: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class _MessagesView:
    """Dual-mode view: iterable list *and* callable with optional session_id."""

    def __init__(self, stm: "ShortTermMemory") -> None:
        self._stm = stm

    def _default_list(self) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self._stm._bucket(_DEFAULT_SID)]

    def __call__(
        self,
        session_id: str = _DEFAULT_SID,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        items = [m.to_dict() for m in self._stm._bucket(session_id)]
        return items[-limit:] if limit is not None else items

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self._default_list())

    def __len__(self) -> int:
        return len(self._stm._bucket(_DEFAULT_SID))

    def __le__(self, other: Any) -> bool:
        return len(self) <= other

    def __getitem__(self, idx: Any) -> Any:
        return self._default_list()[idx]

    def __repr__(self) -> str:
        return repr(self._default_list())


class ShortTermMemory:
    """FIFO ring buffer supporting both session-based and single-session APIs."""

    def __init__(self, max_messages: int = 50) -> None:
        self.max_messages = max_messages
        self._sessions: Dict[str, Deque[Message]] = {}

    def _bucket(self, session_id: str) -> Deque[Message]:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_messages)
        return self._sessions[session_id]

    def add(
        self,
        session_id_or_role: str,
        role_or_content: str,
        content: Optional[str] = None,
        **meta: str,
    ) -> Message:
        if content is None:
            sid, role, text = _DEFAULT_SID, session_id_or_role, role_or_content
        else:
            sid, role, text = session_id_or_role, role_or_content, content
        msg = Message(role=role, content=text, meta=meta)
        self._bucket(sid).append(msg)
        return msg

    @property
    def messages(self) -> _MessagesView:
        return _MessagesView(self)

    def get_recent(self, n: int) -> List[Dict[str, str]]:
        items = list(self._bucket(_DEFAULT_SID))
        return [m.to_dict() for m in items[-n:]]

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)


class SessionMemory:
    """Multi-session wrapper used by MemoryManager."""

    def __init__(self, max_messages: int = 50) -> None:
        self._max = max_messages
        self._sessions: Dict[str, ShortTermMemory] = {}

    def _get(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory(self._max)
        return self._sessions[session_id]

    def add(self, session_id: str, role: str, content: str, **meta: str) -> Message:
        return self._get(session_id).add(role, content, **meta)

    def history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        return self._get(session_id).messages(limit=limit)

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)
