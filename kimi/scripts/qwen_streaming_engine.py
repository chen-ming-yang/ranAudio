#!/usr/bin/env python3
"""L1 streaming ASR engine: Qwen3-ASR-1.7B (vLLM) → PartialHypothesis.

Maps Qwen streaming state to PartialHypothesis {text, stable_len, is_final}.
stable_len follows unfixed_token_num rollback (prefix tokens = stable).
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_MODEL = os.environ.get(
    "KIMI_QWEN_ASR",
    "/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B",
)


@dataclass
class PartialHypothesis:
    type: str  # "partial" | "final"
    session_id: str
    text: str
    stable_len: int
    audio_ms: int
    is_final: bool
    engine: str
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "text": self.text,
            "stable_len": int(self.stable_len),
            "audio_ms": int(self.audio_ms),
            "is_final": bool(self.is_final),
            "engine": self.engine,
            "language": self.language or "",
        }


def _lcp_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compute_stable_len(
    text: str,
    *,
    tokenizer,
    raw_decoded: str,
    chunk_id: int,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    is_final: bool,
) -> int:
    """Stable prefix length in characters (Unicode codepoints / Python str chars)."""
    if not text:
        return 0
    if is_final:
        return len(text)
    if chunk_id < int(unfixed_chunk_num):
        return 0
    # Prefer parsed text tokenization so stable_len aligns with displayed text.
    try:
        ids = tokenizer.encode(text)
    except Exception:
        ids = tokenizer.encode(raw_decoded or text)
    k = max(0, int(unfixed_token_num))
    end_idx = max(0, len(ids) - k)
    if end_idx <= 0:
        return 0
    try:
        prefix = tokenizer.decode(ids[:end_idx])
    except Exception:
        prefix = text[: max(0, len(text) - k)]
    # Align decode artifacts to displayed text via LCP.
    return min(len(text), _lcp_len(prefix, text))


class QwenStreamingEngine:
    """One vLLM Qwen3-ASR model; sessions are single-stream (API constraint)."""

    ENGINE_NAME = "qwen3-asr-1.7b"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        gpu_memory_utilization: float = 0.35,
        max_new_tokens: int = 32,
        max_model_len: int = 8192,
        enforce_eager: bool = True,
        language: str = "Chinese",
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 1.0,
    ) -> None:
        self.model_path = model_path
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_new_tokens = int(max_new_tokens)
        self.max_model_len = int(max_model_len)
        self.enforce_eager = bool(enforce_eager)
        self.language = language
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)
        self.chunk_size_sec = float(chunk_size_sec)
        self.model = None
        self._tokenizer = None
        self._lock_session: Optional[str] = None  # single active stream
        self._state = None
        self._session_id: Optional[str] = None
        self._last_text = ""
        self.ready = False
        self.load_error = ""

    def load(self) -> None:
        from qwen_asr import Qwen3ASRModel

        t0 = time.time()
        # enforce_eager: skip torch.compile/triton (needs system C compiler)
        self.model = Qwen3ASRModel.LLM(
            model=self.model_path,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_new_tokens=self.max_new_tokens,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
        )
        # tokenizer lives on processor
        self._tokenizer = self.model.processor.tokenizer
        self.ready = True
        self.load_error = ""
        print(
            f"[l1-engine] loaded {self.model_path} in {time.time() - t0:.1f}s "
            f"chunk={self.chunk_size_sec}s unfixed_tok={self.unfixed_token_num}",
            flush=True,
        )

    def _hyp(self, *, is_final: bool) -> PartialHypothesis:
        assert self._state is not None and self._session_id is not None
        text = (self._state.text or "").strip()
        # Monotonic soft lock: never shrink stable vs previous emit unless LCP forces it
        stable = compute_stable_len(
            text,
            tokenizer=self._tokenizer,
            raw_decoded=getattr(self._state, "_raw_decoded", "") or "",
            chunk_id=int(self._state.chunk_id),
            unfixed_chunk_num=self.unfixed_chunk_num,
            unfixed_token_num=self.unfixed_token_num,
            is_final=is_final,
        )
        if not is_final and self._last_text:
            # If model rolled back displayed text, shrink via LCP with last text
            lcp = _lcp_len(self._last_text, text)
            if lcp < len(self._last_text):
                stable = min(stable, lcp)
        audio_ms = int(1000.0 * float(self._state.audio_accum.shape[0]) / SAMPLE_RATE)
        self._last_text = text
        return PartialHypothesis(
            type="final" if is_final else "partial",
            session_id=self._session_id,
            text=text,
            stable_len=stable,
            audio_ms=audio_ms,
            is_final=is_final,
            engine=self.ENGINE_NAME,
            language=getattr(self._state, "language", "") or "",
        )

    def start_session(
        self,
        session_id: Optional[str] = None,
        *,
        language: Optional[str] = None,
    ) -> str:
        if not self.ready or self.model is None:
            raise RuntimeError(self.load_error or "engine not loaded")
        if self._lock_session is not None:
            # Force-close previous (single-stream API)
            try:
                self.finish_session()
            except Exception:
                self._lock_session = None
                self._state = None
        sid = session_id or uuid.uuid4().hex[:12]
        lang = language if language is not None else self.language
        self._state = self.model.init_streaming_state(
            context="",
            language=lang or None,
            unfixed_chunk_num=self.unfixed_chunk_num,
            unfixed_token_num=self.unfixed_token_num,
            chunk_size_sec=self.chunk_size_sec,
        )
        self._session_id = sid
        self._lock_session = sid
        self._last_text = ""
        return sid

    def feed_pcm16k(self, pcm: np.ndarray) -> Optional[PartialHypothesis]:
        """Feed arbitrary-length 16k mono float32/int16. Returns hyp if text changed."""
        if self._state is None or self.model is None:
            raise RuntimeError("no active session")
        prev = self._state.text
        prev_chunk = self._state.chunk_id
        self.model.streaming_transcribe(pcm, self._state)
        if self._state.chunk_id == prev_chunk and (self._state.text or "") == (prev or ""):
            return None  # still buffering toward chunk_size
        return self._hyp(is_final=False)

    def finish_session(self) -> PartialHypothesis:
        if self._state is None or self.model is None or self._session_id is None:
            raise RuntimeError("no active session")
        self.model.finish_streaming_transcribe(self._state)
        hyp = self._hyp(is_final=True)
        self._lock_session = None
        self._state = None
        # keep session_id on hyp; clear local
        self._session_id = None
        self._last_text = ""
        return hyp


class FakeStreamingEngine:
    """UI smoke engine: echoes growing Chinese-like placeholder from energy."""

    ENGINE_NAME = "fake"

    def __init__(self, **_kwargs: Any) -> None:
        self.ready = True
        self.load_error = ""
        self._session_id: Optional[str] = None
        self._n = 0
        self._chars = list("今天天气不错适合出门散步聊天")
        self.chunk_size_sec = 0.4

    def load(self) -> None:
        self.ready = True

    def start_session(self, session_id: Optional[str] = None, **_kw: Any) -> str:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._n = 0
        return self._session_id

    def feed_pcm16k(self, pcm: np.ndarray) -> Optional[PartialHypothesis]:
        if self._session_id is None:
            raise RuntimeError("no active session")
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if x.size < 160:  # <10ms
            return None
        self._n += 1
        n_chars = min(len(self._chars), 1 + self._n // 3)
        text = "".join(self._chars[:n_chars])
        hold = 2 if n_chars > 2 else 0
        stable = max(0, n_chars - hold)
        audio_ms = int(self._n * 40)  # rough
        return PartialHypothesis(
            type="partial",
            session_id=self._session_id,
            text=text,
            stable_len=stable,
            audio_ms=audio_ms,
            is_final=False,
            engine=self.ENGINE_NAME,
        )

    def finish_session(self) -> PartialHypothesis:
        if self._session_id is None:
            raise RuntimeError("no active session")
        n_chars = min(len(self._chars), max(1, 1 + self._n // 3))
        text = "".join(self._chars[:n_chars])
        hyp = PartialHypothesis(
            type="final",
            session_id=self._session_id,
            text=text,
            stable_len=len(text),
            audio_ms=int(self._n * 40),
            is_final=True,
            engine=self.ENGINE_NAME,
        )
        self._session_id = None
        self._n = 0
        return hyp
