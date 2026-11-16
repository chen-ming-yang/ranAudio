"""Pluggable FastLM slot: one event contract, several backends.

The L2 coordinator only depends on `FastLMBackend`. `RecordedBackend` replays
real historical GPU outputs AND (optionally) the measured per-stage latencies,
so the whole streaming orchestration is runnable and benchmarkable without a
GPU while still reflecting the real model costs from 2026-11-02.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from .events import Clock, StreamEvent
from .stream_protocol import NOFC, IncrementalParser

BASELINE_JSON = Path("/tmp/step_route_results.json")


@runtime_checkable
class FastLMBackend(Protocol):
    name: str

    def route_stream(self, wav: str | Path, request_id: str) -> Iterator[StreamEvent]: ...

    def reinject_stream(
        self, speak: str, wav: Optional[str | Path], request_id: str
    ) -> Iterator[StreamEvent]: ...


def stream_text(
    text: str,
    request_id: str,
    *,
    chunk_chars: int = 4,
    clock: Clock | None = None,
    delay_ms: float = 0.0,
    t_offset_ms: float = 0.0,
) -> Iterator[StreamEvent]:
    """Feed a finished string through the incremental parser as token deltas.

    `t_offset_ms` shifts event timestamps to account for time already spent in
    prefill/encode before the first token (replayed model latency).
    """
    parser = IncrementalParser(request_id, clock or Clock())

    def shift(events):
        return [replace(e, t_ms=round(e.t_ms + t_offset_ms, 3)) for e in events] if t_offset_ms else events

    for i in range(0, len(text), max(1, chunk_chars)):
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        yield from shift(parser.feed(text[i : i + chunk_chars]))
    yield from shift(parser.finish())


def _lookup(mapping: dict[str, Any], stem: str) -> Any | None:
    """Exact stem, or a stem carrying the original name (VAD temp wavs)."""
    if stem in mapping:
        return mapping[stem]
    for key, value in mapping.items():
        if key and key in stem:
            return value
    return None


class RecordedBackend:
    """Offline replay of `/tmp/step_route_results.json` (real 2026-11-02 outputs).

    `simulate_latency=True` sleeps the measured fast/reinject durations before
    emitting deltas, which models prefill cost without loading a model.
    """

    name = "recorded"

    def __init__(
        self,
        route_texts: dict[str, str] | None = None,
        reinject_texts: dict[str, str] | None = None,
        speaks: dict[str, str] | None = None,
        *,
        latencies: dict[str, dict[str, float]] | None = None,
        chunk_chars: int = 4,
        token_delay_ms: float = 0.0,
        simulate_latency: bool = False,
    ) -> None:
        self.route_texts = route_texts or {}
        self.reinject_texts = reinject_texts or {}
        self.speaks = speaks or {}
        self.latencies = latencies or {}
        self.chunk_chars = chunk_chars
        self.token_delay_ms = token_delay_ms
        self.simulate_latency = simulate_latency

    @classmethod
    def from_baseline(cls, path: str | Path = BASELINE_JSON, **kwargs) -> "RecordedBackend":
        p = Path(path)
        obj = cls(**kwargs)
        if not p.is_file():
            return obj
        for row in json.loads(p.read_text(encoding="utf-8")):
            stem = Path(str(row.get("wav") or "")).stem
            if not stem:
                continue
            if row.get("raw_fast"):
                obj.route_texts[stem] = str(row["raw_fast"])
            if row.get("raw_reinject"):
                obj.reinject_texts[stem] = str(row["raw_reinject"])
            if row.get("slow_speak"):
                obj.speaks[stem] = str(row["slow_speak"])
            if row.get("latency"):
                obj.latencies[stem] = {
                    k: float(v) for k, v in row["latency"].items() if isinstance(v, (int, float))
                }
        return obj

    def sleep_for(self, stem: str, key: str) -> float:
        """Replay one measured stage duration. Returns seconds slept."""
        if not self.simulate_latency:
            return 0.0
        info = _lookup(self.latencies, stem) or {}
        seconds = max(0.0, float(info.get(key) or 0.0))
        if seconds > 0:
            time.sleep(seconds)
        return seconds

    def _stream(
        self, text: str, request_id: str, *, t_offset_ms: float = 0.0
    ) -> Iterator[StreamEvent]:
        yield from stream_text(
            text,
            request_id,
            chunk_chars=self.chunk_chars,
            delay_ms=self.token_delay_ms,
            t_offset_ms=t_offset_ms,
        )

    def route_stream(self, wav: str | Path, request_id: str) -> Iterator[StreamEvent]:
        stem = Path(str(wav)).stem
        prefill_s = self.sleep_for(stem, "fast_s")  # before the first token
        text = _lookup(self.route_texts, stem) or f"{NOFC}我没有听清，请再说一遍。"
        yield from self._stream(text, request_id, t_offset_ms=prefill_s * 1000.0)

    def reinject_stream(
        self, speak: str, wav: Optional[str | Path], request_id: str
    ) -> Iterator[StreamEvent]:
        stem = Path(str(wav)).stem if wav else ""
        prefill_s = self.sleep_for(stem, "reinject_s")
        text = _lookup(self.reinject_texts, stem) or f"{NOFC}{speak}"
        yield from self._stream(text, request_id, t_offset_ms=prefill_s * 1000.0)
