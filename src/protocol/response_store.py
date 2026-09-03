"""Bounded, process-local store of verified conversation state for ``/v1/responses``.

Issue #27, Option 1. An OpenAI-style client that sends only a new ``input`` plus a
``previous_response_id`` is otherwise stateless between turns, so referent
follow-ups ("his role there?", "a su experiencia profesional") lose the topic and
source they point at. This store lets the adapter resolve that ID back to the
compact verified state the core agent already produces.

What it deliberately is not: a transcript database. Entries are
:class:`~src.agent.contracts.ConversationState` objects, whose identifier fields
are regex-constrained to reject whitespace, so no message or answer prose can be
persisted here. Entries expire after a TTL and the store is capacity-capped;
overflow evicts the oldest entry. Everything is lost on process restart, which
fails closed: an unresolvable ID becomes a machine-readable error and the client
may fall back to resending history in ``input``.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from src.agent.contracts import ConversationState


@dataclass(frozen=True)
class _Entry:
    """One stored snapshot with the clock reading taken when it was written."""

    state: ConversationState
    stored_at: float


class ResponseStateStore:
    """Thread-safe TTL + capacity map from an opaque ``resp_*`` ID to verified state."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()

    def put(self, response_id: str, state: ConversationState) -> None:
        """Store or refresh ``state`` under ``response_id``, then enforce both bounds."""
        now = self._clock()
        with self._lock:
            self._drop_expired(now)
            self._entries[response_id] = _Entry(state=state, stored_at=now)
            self._entries.move_to_end(response_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get(self, response_id: str) -> ConversationState | None:
        """Return the live state for ``response_id`` or ``None`` if unknown or expired."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(response_id)
            if entry is None:
                return None
            if now - entry.stored_at >= self._ttl:
                del self._entries[response_id]
                return None
            return entry.state

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _drop_expired(self, now: float) -> None:
        """Remove every entry past its TTL. Caller must hold the lock."""
        expired = [
            response_id
            for response_id, entry in self._entries.items()
            if now - entry.stored_at >= self._ttl
        ]
        for response_id in expired:
            del self._entries[response_id]
