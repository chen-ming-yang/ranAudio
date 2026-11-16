"""Shared event / utterance contracts for the streaming FastLM runtime.

Pure data + clock helpers: no torch, no model, no GPU. All backends and the L2
coordinator speak this one event model.
"""

from __future__ import annotations

import re
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

EventKind = Literal[
    "route_first",
    "transcript_delta",
    "body_delta",
    "route_final",
    "speak",
    "done",
    "cancelled",
]
Route = Optional[Literal["fc", "nofc", "other"]]


@dataclass(frozen=True)
class StreamEvent:
    kind: EventKind
    request_id: str
    t_ms: float
    text: str = ""
    route: Route = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Utterance:
    """One VAD-segmented utterance. `pcm` is float32 mono 16k."""

    request_id: str
    pcm: np.ndarray
    sr: int = 16000
    start_ms: float = 0.0
    end_ms: float = 0.0
    source: str = ""
    barge_in: bool = False

    @property
    def duration_ms(self) -> float:
        return float(len(self.pcm)) / float(self.sr) * 1000.0

    def save_wav(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(self.source).stem)[:48] if self.source else ""
        path = out / f"utt_{stem + '_' if stem else ''}{self.request_id}.wav"
        pcm = np.clip(np.asarray(self.pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
        data = (pcm * 32767.0).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.sr))
            w.writeframes(data)
        return path


class Clock:
    """Monotonic millisecond clock relative to one utterance."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000.0, 3)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def segment_latency(events: list[StreamEvent], kind: EventKind) -> Optional[float]:
    """First event of `kind`, in ms from utterance start."""
    for ev in events:
        if ev.kind == kind:
            return ev.t_ms
    return None
