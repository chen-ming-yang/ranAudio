"""Offline orchestration: FastLM route → optional SlowLM → backend reinject → user_text."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .fastlm import StepFastLM
from .protocol_step import parse_s2_output, strip_nofc_prefix
from .slowlm import SlowLMClient


@dataclass
class TurnResult:
    request_id: str
    wav: str
    route: str
    raw_fast: str
    transcript: Optional[str] = None
    fast_reply: Optional[str] = None
    slow_speak: Optional[str] = None
    raw_reinject: Optional[str] = None
    final_reply: Optional[str] = None
    user_text: Optional[str] = None  # TTS 出口文本；SlowLM 不得直通此字段
    latency: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Streaming-only fields. Added at the end so existing to_dict() keys stay stable.
    route_first_ms: Optional[float] = None
    emit_ms: Optional[float] = None
    first_speak_ms: Optional[float] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    fallback: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepRoutePipeline:
    """Demo pipeline. Emit is logically fire-and-forget; offline run is sequential."""

    def __init__(
        self,
        fast: StepFastLM,
        slow: Optional[SlowLMClient] = None,
        *,
        reinject: bool = True,
        reinject_with_wav: bool = True,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.reinject = reinject
        self.reinject_with_wav = reinject_with_wav

    def run(self, wav: str | Path) -> TurnResult:
        wav_s = str(wav)
        rid = uuid.uuid4().hex[:12]
        t_all = time.time()
        raw_fast, t_fast = self.fast.route(wav_s)
        parsed = parse_s2_output(raw_fast)
        result = TurnResult(
            request_id=rid,
            wav=wav_s,
            route=parsed["route"],
            raw_fast=raw_fast,
            latency={"fast_s": round(t_fast, 3)},
        )

        if parsed["route"] == "nofc":
            result.fast_reply = parsed["body"]
            result.final_reply = parsed["body"]
            result.user_text = parsed["body"]
            result.latency["total_s"] = round(time.time() - t_all, 3)
            result.notes.append("fast_loop: NOFC → user_text (no SlowLM)")
            return result

        if parsed["route"] != "fc":
            # Degraded: treat whole string as local reply so demo still has an outlet.
            result.fast_reply = parsed["body"]
            result.final_reply = parsed["body"]
            result.user_text = parsed["body"]
            result.notes.append("route=other; degraded to local user_text")
            result.latency["total_s"] = round(time.time() - t_all, 3)
            return result

        # FC+BASE: dispatch
        result.transcript = parsed["transcript"]
        result.notes.append("slow_loop: FC+BASE emit (logical fire-and-forget)")

        if self.slow is None:
            result.notes.append("SlowLM not attached; stop after dispatch")
            result.latency["total_s"] = round(time.time() - t_all, 3)
            return result

        t1 = time.time()
        speak = self.slow.speak(result.transcript or "")
        result.slow_speak = speak
        result.latency["slow_s"] = round(time.time() - t1, 3)

        if not speak:
            result.notes.append("slow_speak empty → no_op (no TTS)")
            result.latency["total_s"] = round(time.time() - t_all, 3)
            return result

        if not self.reinject:
            # Explicitly do NOT put slow_speak into user_text (铁律 3).
            result.notes.append("reinject disabled; slow_speak kept off TTS outlet")
            result.latency["total_s"] = round(time.time() - t_all, 3)
            return result

        t2 = time.time()
        reinject_wav = wav_s if self.reinject_with_wav else None
        raw_re, _ = self.fast.reinject(speak, wav=reinject_wav)
        result.raw_reinject = raw_re
        result.latency["reinject_s"] = round(time.time() - t2, 3)
        final = strip_nofc_prefix(raw_re) if raw_re else ""
        result.final_reply = final
        result.user_text = final
        result.notes.append("backend reinject → NOFC 拼接稿 → user_text")
        result.latency["total_s"] = round(time.time() - t_all, 3)
        return result


class RealtimePipeline:
    """Streaming entry point. Delegates to `coordinator.RealtimeRunner`.

    `StepRoutePipeline` above is the untouched synchronous baseline; all new
    behaviour lives behind an explicit `mode`.
    """

    def __init__(
        self,
        mode: str = "offline-events",
        wavs: list[str] | None = None,
        out: str | Path = "/tmp/realtime_results.json",
        *,
        slow: Optional[Any] = None,
        user_speaking: bool = False,
        use_vad: bool = False,
        independent_turns: bool = True,
    ) -> None:
        self.mode = mode
        self.wavs = list(wavs or [])
        self.out = out
        self.slow = slow
        self.user_speaking = user_speaking
        self.use_vad = use_vad
        self.independent_turns = independent_turns
        self.speaks: list[dict[str, Any]] = []
        self.store: Optional[Any] = None

    def run_all(self) -> list[dict[str, Any]]:
        from .coordinator import RealtimeRunner

        runner = RealtimeRunner(
            self.mode,
            self.wavs,
            self.out,
            slow=self.slow,
            user_speaking=self.user_speaking,
            use_vad=self.use_vad,
            independent_turns=self.independent_turns,
        )
        self.speaks = runner.speaks
        self.store = runner.store
        return runner.run_all()
