"""Incremental S2 protocol parser and backend fact-fidelity guard.

`parse_s2_output` in protocol_step.py works on a finished string. This module
works on a token stream: it tolerates half-emitted special tokens, emits the
route as soon as the first route tag is complete, and streams body/transcript
deltas without waiting for the full generation.

Pure Python: falls back to local literals when `ranAudio.protocol` (which
imports torch) is unavailable, so no-GPU unit tests stay runnable.
"""

from __future__ import annotations

import re
from typing import Any

from .events import Clock, Route, StreamEvent

try:  # pragma: no cover - exercised in the model environments
    from ranAudio.protocol import (
        BACKEND_END,
        BACKEND_START,
        BASE,
        FC,
        NOFC,
        entity_fidelity,
    )
except Exception:  # no torch / no package on sys.path
    FC = "<|FC|>"
    NOFC = "<|NOFC|>"
    BASE = "<|BASE|>"
    BACKEND_START = "<backend>"
    BACKEND_END = "</backend>"
    _DIGIT_RE = re.compile(r"\d+")

    def entity_fidelity(ref: str, hyp: str):
        needles = _DIGIT_RE.findall(ref or "")
        if not needles:
            return None
        return sum(1 for n in needles if n in (hyp or "")) / len(needles)


START_TAGS: tuple[str, ...] = (FC + BASE, FC, NOFC, "<FC>", "<NOFC>")
STOP_TAGS: tuple[str, ...] = (
    NOFC,
    FC,
    BASE,
    BACKEND_START,
    BACKEND_END,
    "<FC>",
    "<NOFC>",
    "<BASE>",
    "</FC>",
    "</NOFC>",
    "</BASE>",
    "<tool_call>",
    "<|EOT|>",
    "<|im_end|>",
    "<|im_start|>",
)
ALL_TAGS: tuple[str, ...] = tuple(dict.fromkeys(START_TAGS + STOP_TAGS))
MAX_TAG_LEN = max(len(t) for t in ALL_TAGS)


def _is_tag_prefix(text: str, tags: tuple[str, ...] = ALL_TAGS) -> bool:
    return bool(text) and any(t.startswith(text) and len(text) < len(t) for t in tags)


def _longest_partial_suffix(text: str, tags: tuple[str, ...] = ALL_TAGS) -> int:
    for k in range(min(len(text), MAX_TAG_LEN), 0, -1):
        if _is_tag_prefix(text[-k:], tags):
            return k
    return 0


def normalize_tags(text: str) -> str:
    """Normalise Step's pipe-less slips on a finished string."""
    raw = (text or "").replace("\u200b", "").replace("\ufeff", "")
    raw = raw.replace("<FC>", FC).replace("</FC>", "")
    raw = raw.replace("<NOFC>", NOFC).replace("</NOFC>", "")
    raw = raw.replace("<BASE>", BASE).replace("</BASE>", "")
    return raw


class IncrementalParser:
    """Token stream -> StreamEvent list. One instance per utterance/forward."""

    def __init__(self, request_id: str, clock: Clock | None = None) -> None:
        self.request_id = request_id
        self.clock = clock or Clock()
        self.raw = ""
        self.state = "prefix"  # prefix | fc_wait_base | transcript | body
        self.route: Route = None
        self.full_text = ""
        self.stopped = False
        self.finished = False

    def _event(self, kind, *, text: str = "", route: Route = None, **meta) -> StreamEvent:
        return StreamEvent(
            kind=kind,
            request_id=self.request_id,
            t_ms=self.clock.ms(),
            text=text,
            route=route,
            meta=meta,
        )

    def _emit_delta(self, chunk: str, out: list[StreamEvent]) -> None:
        if not chunk:
            return
        self.full_text += chunk
        kind = "transcript_delta" if self.state == "transcript" else "body_delta"
        out.append(self._event(kind, text=chunk, route=self.route))

    def feed(self, text: str) -> list[StreamEvent]:
        if self.finished or self.stopped or not text:
            return []
        self.raw += text
        return self._drain()

    def _drain(self) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        while self.raw and not self.stopped:
            if self.state == "prefix":
                matched = next((t for t in START_TAGS if self.raw.startswith(t)), None)
                if matched:
                    self.raw = self.raw[len(matched) :]
                    if matched == FC + BASE:
                        self.route, self.state = "fc", "transcript"
                    elif matched in (FC, "<FC>"):
                        self.route, self.state = "fc", "fc_wait_base"
                    else:
                        self.route, self.state = "nofc", "body"
                    out.append(self._event("route_first", route=self.route, tag=matched))
                    continue
                if _is_tag_prefix(self.raw):
                    return out
                # Soft recovery: drop leading chatter until a tag appears.
                hits = [(self.raw.find(t), t) for t in ALL_TAGS if self.raw.find(t) >= 0]
                if hits:
                    i, tag = min(hits, key=lambda x: x[0])
                    if i == 0 and tag not in START_TAGS:
                        self.raw = self.raw[len(tag) :]
                        continue
                    self.raw = self.raw[i:]
                    continue
                keep = _longest_partial_suffix(self.raw)
                self.raw = self.raw[-keep:] if keep else ""
                return out

            if self.state == "fc_wait_base":
                if self.raw.startswith(BASE) or self.raw.startswith("<BASE>"):
                    n = len(BASE) if self.raw.startswith(BASE) else len("<BASE>")
                    self.raw = self.raw[n:]
                    self.state = "transcript"
                    continue
                if _is_tag_prefix(self.raw, (BASE, "<BASE>")):
                    return out
                # `<|FC|>` without BASE: degrade like parse_s2_output route=other.
                self.route, self.state = "other", "body"
                continue

            # transcript / body
            stop = next((t for t in STOP_TAGS if self.raw.startswith(t)), None)
            if stop:
                self.stopped = True
                self.raw = ""
                return out
            if _is_tag_prefix(self.raw, STOP_TAGS):
                return out
            keep = _longest_partial_suffix(self.raw, STOP_TAGS)
            emit = self.raw[:-keep] if keep else self.raw
            self.raw = self.raw[-keep:] if keep else ""
            self._emit_delta(emit, out)
            return out
        return out

    def finish(self) -> list[StreamEvent]:
        if self.finished:
            return []
        self.finished = True
        out: list[StreamEvent] = []
        if not self.stopped and self.raw and self.state in ("transcript", "body"):
            keep = _longest_partial_suffix(self.raw, STOP_TAGS)
            emit = self.raw[:-keep] if keep else self.raw
            self._emit_delta(emit, out)
        self.raw = ""
        route: Route = self.route or "other"
        out.append(
            self._event(
                "route_final",
                text=self.full_text,
                route=route,
                tag_compliant=route != "other",
            )
        )
        out.append(self._event("done", route=route))
        return out


def _norm(text: str) -> str:
    """Whitespace-insensitive verbatim check (TTS text often drops spaces)."""
    return "".join((text or "").split())


def fact_ok(speak: str, reply: str) -> bool:
    """Backend facts must survive verbatim; digits alone are not enough."""
    s = (speak or "").strip()
    return bool(s) and _norm(s) in _norm(reply)


def guard_speak(speak: str, reply: str, *, prefix: str = "查到了，") -> tuple[str, dict[str, Any]]:
    """Return (playable text, diagnostics). Falls back to structured splicing.

    A generative copy can silently turn a demo placeholder into a confident
    number (observed in /tmp/step_route_results.json), so an L2 check is
    mandatory before anything reaches the single TTS outlet.
    """
    s = (speak or "").strip()
    r = (reply or "").strip()
    fid = entity_fidelity(s, r)
    if fact_ok(s, r):
        return r, {"fallback": False, "fidelity": fid, "reason": "verbatim"}
    text = f"{prefix}{s}" if prefix else s
    return text, {"fallback": True, "fidelity": fid, "reason": "fact_mismatch"}
