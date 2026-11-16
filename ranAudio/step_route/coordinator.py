"""L2 realtime coordinator: fire-and-forget slow loop, one TTS outlet.

Keeps STEP_ROUTE.md's four invariants:
  1. FC+BASE read -> emit immediately, no filler.
  2. Emit does not await SlowLM (thread pool + store callback).
  3. SlowLM never writes TTS directly; it lands as a `<backend>` reinject event.
  4. Only FastLM NOFC text (after a fidelity guard) reaches `user_text`.

The synchronous StepRoutePipeline stays untouched as the baseline.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

from .events import Clock, StreamEvent, Utterance, new_request_id, segment_latency
from .fastlm_base import RecordedBackend
from .session_store import SessionStore
from .stream_protocol import guard_speak
from .vad import VADConfig, VADStream, load_wav_mono16k

MODES = ("offline-events", "step-stream", "minicpm-probe")


class RealtimeRunner:
    def __init__(
        self,
        mode: str = "offline-events",
        wavs: list[str] | None = None,
        out: str | Path = "/tmp/realtime_results.json",
        *,
        slow: Any = None,
        user_speaking: bool = False,
        use_vad: bool = False,
        independent_turns: bool = True,
        tmp_dir: str | Path = "/tmp/realtime_utts",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode={mode}; choices={MODES}")
        self.mode = mode
        self.wavs = [str(w) for w in (wavs or [])]
        self.out = Path(out)
        self.slow = slow
        self.user_speaking = bool(user_speaking)
        self.use_vad = bool(use_vad)
        self.independent_turns = bool(independent_turns)
        self.tmp_dir = Path(tmp_dir)
        self.store = SessionStore(independent_turns=self.independent_turns)
        self.backend: Any = None
        self.speaks: list[dict[str, Any]] = []
        self._last_rid = ""
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    # ---- setup ---------------------------------------------------------- #
    def _build_backend(self) -> Any:
        if self.mode == "offline-events":
            return RecordedBackend.from_baseline(simulate_latency=True)
        if self.mode == "step-stream":
            from .fastlm_stream import StepStreamBackend

            backend = StepStreamBackend()
            backend.load()
            return backend
        from .fastlm_minicpm import MiniCPMProbeBackend

        backend = MiniCPMProbeBackend()
        backend.start(timeout=300.0)
        return backend

    def _close_backend(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # ---- single TTS outlet ---------------------------------------------- #
    def _sink(self, request_id: str, text: str, row: dict[str, Any], *, source: str) -> None:
        if not (text or "").strip():
            return
        with self._lock:
            self.speaks.append(
                {
                    "request_id": request_id,
                    "text": text,
                    "source": source,
                    "t_ms": row.get("latency", {}).get("first_speak_ms"),
                }
            )
        # No audio device / edge-tts in the prototype: text sink is the outlet.

    def _slow_fn(self, transcript: str, request_id: str, stem: str) -> str:
        if self.slow is not None:
            try:
                return self.slow.speak(transcript, request_id=request_id)
            except TypeError:  # older client signature
                return self.slow.speak(transcript)
        recorded = getattr(self.backend, "speaks", {}) or {}
        sleep_for = getattr(self.backend, "sleep_for", None)
        if callable(sleep_for):
            sleep_for(stem, "slow_s")  # replay measured SlowLM latency
        return recorded.get(stem) or f"（SlowLM 未挂载）已收到：{transcript}"

    # ---- per utterance -------------------------------------------------- #
    def _utterances(self, wav: str) -> list[Utterance]:
        pcm, sr = load_wav_mono16k(wav)
        if not self.use_vad:
            return [
                Utterance(
                    request_id=new_request_id(),
                    pcm=pcm,
                    sr=sr,
                    end_ms=len(pcm) / float(sr) * 1000.0,
                    source=wav,
                )
            ]
        stream = VADStream(sr=sr, source=wav)
        utts = stream.feed(pcm) + stream.flush()
        return utts or [
            Utterance(
                request_id=new_request_id(),
                pcm=pcm,
                sr=sr,
                end_ms=len(pcm) / float(sr) * 1000.0,
                source=wav,
            )
        ]

    def _submit_slow(
        self,
        rid: str,
        transcript: str,
        stem: str,
        wav_path: Path,
        row: dict[str, Any],
        clock: Clock,
    ) -> None:
        assert self._executor is not None

        def job() -> None:
            t_slow = time.perf_counter()
            try:
                speak = self._slow_fn(transcript, rid, stem)
            except Exception as exc:
                row["notes"].append(f"SlowLM error: {type(exc).__name__}: {exc}")
                row["latency"]["total_s"] = round(clock.ms() / 1000.0, 3)
                return
            row["slow_speak"] = speak
            row["latency"]["slow_s"] = round(time.perf_counter() - t_slow, 3)
            if not (speak or "").strip():
                row["notes"].append("slow_speak empty → no_op (no TTS)")
                return
            if not self.store.set_pending(rid, speak):
                row["notes"].append("slow result dropped: stale/cancelled request_id")
                row["latency"]["drop_ms"] = clock.ms()
                return
            decision = self.store.arbitrate(rid, user_speaking=self.user_speaking)
            if decision == "hold":
                row["notes"].append("hold: user speaking; reinject deferred to endpoint")
                decision = "speak"  # offline prototype: next endpoint arrives immediately
                row["notes"].append("hold resolved at next endpoint (offline simulation)")
            if decision == "drop":
                row["notes"].append("arbitration drop")
                return
            t_re = time.perf_counter()
            try:
                events = list(self.backend.reinject_stream(speak, wav_path, rid))
            except Exception as exc:
                row["notes"].append(f"reinject error: {type(exc).__name__}: {exc}")
                return
            row["latency"]["reinject_s"] = round(time.perf_counter() - t_re, 3)
            final = next((e for e in events if e.kind == "route_final"), None)
            reply = final.text if final else ""
            prefix = "" if speak.startswith(("已", "好的", "音量已", "温度已", "空调已")) else "查到了，"
            playable, meta = guard_speak(speak, reply, prefix=prefix)
            row["raw_reinject"] = reply
            row["final_reply"] = playable
            row["user_text"] = playable
            row["fallback"] = meta
            row["latency"]["first_speak_ms"] = clock.ms()
            row["latency"]["total_s"] = round(clock.ms() / 1000.0, 3)
            row["notes"].append("backend reinject → fidelity guard → single TTS outlet")
            self.store.append_history(playable)
            self._sink(rid, playable, row, source="reinject")

        self._executor.submit(job)

    def _run_utterance(self, utt: Utterance, wav_path: Path, source_wav: str) -> dict[str, Any]:
        clock = Clock()
        rid = utt.request_id
        if not self.independent_turns and self._last_rid and self._last_rid != rid:
            cancel = getattr(self.slow, "cancel", None)
            if callable(cancel):
                try:
                    cancel(self._last_rid)
                except Exception:
                    pass
        self._last_rid = rid
        self.store.begin(rid, "", cancel_previous=not self.independent_turns)

        row: dict[str, Any] = {
            "request_id": rid,
            "wav": source_wav,
            "utt_ms": round(utt.duration_ms, 1),
            "mode": self.mode,
            "route": None,
            "raw_fast": "",
            "transcript": None,
            "fast_reply": None,
            "slow_speak": None,
            "raw_reinject": None,
            "final_reply": None,
            "user_text": None,  # single TTS outlet
            "latency": {},
            "notes": [],
            "events": [],
            "fallback": None,
        }
        t_route = time.perf_counter()
        events: list[StreamEvent] = []
        try:
            events = list(self.backend.route_stream(wav_path, rid))
        except Exception as exc:
            row["notes"].append(f"FastLM error: {type(exc).__name__}: {exc}")
            row["latency"]["fast_s"] = round(time.perf_counter() - t_route, 3)
            row["latency"]["total_s"] = round(clock.ms() / 1000.0, 3)
            return row
        row["latency"]["fast_s"] = round(time.perf_counter() - t_route, 3)
        row["events"] = [e.to_dict() for e in events]
        first_ms = segment_latency(events, "route_first")
        if first_ms is not None:
            row["latency"]["route_first_ms"] = round(first_ms, 3)
        final = next((e for e in events if e.kind == "route_final"), None)
        route = (final.route if final else "other") or "other"
        text = final.text if final else ""
        row["route"] = route
        row["raw_fast"] = text

        if route in ("nofc", "other"):
            if route == "other":
                row["notes"].append("route=other; parse degraded, text kept off raw SlowLM path")
            row["fast_reply"] = text
            row["final_reply"] = text
            row["user_text"] = text
            row["latency"]["first_speak_ms"] = clock.ms()
            row["latency"]["fast_loop_s"] = round(clock.ms() / 1000.0, 3)
            row["latency"]["total_s"] = row["latency"]["fast_loop_s"]
            row["notes"].append("fast_loop: NOFC → user_text (no SlowLM)")
            self.store.append_history(text)
            self._sink(rid, text, row, source="fast")
            return row

        row["transcript"] = text
        self.store.begin(rid, text, cancel_previous=not self.independent_turns)
        row["latency"]["emit_ms"] = clock.ms()
        row["latency"]["fast_loop_s"] = round(clock.ms() / 1000.0, 3)
        row["latency"]["total_s"] = row["latency"]["fast_loop_s"]
        row["notes"].append("FC+BASE emit fire-and-forget; listening continues")
        self._submit_slow(rid, text, Path(source_wav).stem, wav_path, row, clock)
        return row

    # ---- entry ---------------------------------------------------------- #
    def run_all(self) -> list[dict[str, Any]]:
        self.backend = self._build_backend()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slow-loop")
        rows: list[dict[str, Any]] = []
        try:
            for wav in self.wavs:
                for utt in self._utterances(wav):
                    wav_path = utt.save_wav(self.tmp_dir)
                    rows.append(self._run_utterance(utt, wav_path, wav))
                    print(
                        f"[realtime] {Path(wav).name} rid={utt.request_id} "
                        f"fast_loop={rows[-1]['latency'].get('fast_loop_s')}s",
                        flush=True,
                    )
        finally:
            self._executor.shutdown(wait=True)
            self._close_backend()
        return rows
