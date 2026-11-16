"""Process-local SessionStore: the single source of truth for L2 context.

Implements STEP_ROUTE.md §5 field contract (session_id / history_text /
last_transcript / request_id / pending_speak / anchor_time). No model KV is
shared between FastLM and SlowLM; only text crosses this boundary.

`independent_turns=True` keeps several in-flight requests alive: batch
benchmarks feed independent utterances back-to-back, and a SlowLM result for
utterance N must NOT be cancelled merely because utterance N+1 started. Real
barge_in uses the default (False), where a new request cancels the old one.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class SessionState:
    session_id: str
    history_text: list[str] = field(default_factory=list)
    last_transcript: str = ""
    request_id: str = ""
    pending_speak: Optional[str] = None
    anchor_time: float = 0.0
    cancelled: set[str] = field(default_factory=set)
    active: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cancelled"] = sorted(self.cancelled)
        return d


class SessionStore:
    def __init__(self, session_id: str = "", *, independent_turns: bool = False) -> None:
        self.independent_turns = bool(independent_turns)
        self.state = SessionState(session_id=session_id or uuid.uuid4().hex[:12])

    def begin(
        self,
        request_id: str,
        transcript: str = "",
        *,
        cancel_previous: Optional[bool] = None,
    ) -> None:
        """Register a request. By default a new request retires the previous."""
        cancel = (not self.independent_turns) if cancel_previous is None else bool(cancel_previous)
        prev = self.state.request_id
        if cancel and prev and prev != request_id:
            self.state.cancelled.add(prev)
            self.state.active.pop(prev, None)
            self.state.pending_speak = None
        self.state.request_id = request_id
        self.state.last_transcript = transcript
        self.state.anchor_time = time.time()
        if request_id not in self.state.cancelled:
            entry = self.state.active.setdefault(request_id, {"transcript": transcript})
            entry["transcript"] = transcript

    def set_pending(self, request_id: str, speak: str) -> bool:
        """SlowLM result arrives. Stale/cancelled ids are rejected."""
        if request_id in self.state.cancelled or request_id not in self.state.active:
            return False
        text = (speak or "").strip()
        if not text:
            return False
        self.state.active[request_id]["pending_speak"] = text
        self.state.pending_speak = text
        return True

    def consume_pending(self, request_id: str) -> Optional[str]:
        entry = self.state.active.get(request_id)
        if not entry or not entry.get("pending_speak"):
            return None
        text = entry.pop("pending_speak")
        if self.state.pending_speak == text:
            self.state.pending_speak = None
        return text

    def cancel(self, request_id: str = "") -> bool:
        """barge_in / explicit cancel: drop the unconsumed backend payload."""
        rid = request_id or self.state.request_id
        if not rid:
            return False
        self.state.cancelled.add(rid)
        entry = self.state.active.get(rid)
        changed = bool(entry and entry.get("pending_speak")) or bool(self.state.pending_speak)
        if entry:
            entry.pop("pending_speak", None)
        self.state.pending_speak = None
        return changed

    def arbitrate(
        self, request_id: str, *, user_speaking: bool
    ) -> Literal["speak", "hold", "drop"]:
        """§4.2: user side wins; stale results drop; hold while user speaks."""
        if request_id in self.state.cancelled:
            return "drop"
        entry = self.state.active.get(request_id)
        if not entry or not entry.get("pending_speak"):
            return "drop"
        if user_speaking:
            return "hold"
        return "speak"

    def append_history(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self.state.history_text.append(t)

    def history(self, n: int = 0) -> list[str]:
        return self.state.history_text[-n:] if n > 0 else list(self.state.history_text)

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()
