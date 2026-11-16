"""Energy/silence VAD for endpointing (L0 stand-in, no model / no weights).

Implements STEP_ROUTE.md §4.1: 400-600 ms tail silence, ~300 ms minimum
utterance, 15 s max_utt, and barge_in buffer reset. Pure numpy + stdlib wave;
no AEC and no real microphone capture in this prototype.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .events import Utterance, new_request_id


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 25
    tail_silence_ms: int = 500
    min_speech_ms: int = 300
    max_utt_ms: int = 15000
    energy_threshold: float = 0.012  # RMS on float32 [-1, 1]
    start_frames: int = 2


def load_wav_mono16k(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav as float32 mono; resample to 16k with linear interpolation."""
    p = Path(path)
    data: np.ndarray
    sr: int
    try:
        import soundfile as sf

        raw, sr = sf.read(str(p), dtype="float32", always_2d=False)
        data = np.asarray(raw, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception:
        with wave.open(str(p), "rb") as w:
            sr = int(w.getframerate())
            n_ch = int(w.getnchannels())
            width = int(w.getsampwidth())
            raw_bytes = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"only 16-bit PCM supported by wave fallback: {p}")
        arr = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0
        data = arr.reshape(-1, n_ch).mean(axis=1) if n_ch > 1 else arr
    data = data.reshape(-1)
    if sr != 16000 and data.size:
        n = int(round(data.size * 16000.0 / sr))
        data = np.interp(
            np.linspace(0.0, data.size - 1, n),
            np.arange(data.size),
            data,
        ).astype(np.float32)
        sr = 16000
    return data.astype(np.float32), sr


def iter_frames(pcm: np.ndarray, frame_len: int) -> Iterator[np.ndarray]:
    for i in range(0, len(pcm), frame_len):
        yield np.asarray(pcm[i : i + frame_len], dtype=np.float32)


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


def energy_vad(
    pcm: np.ndarray,
    sr: int = 16000,
    cfg: VADConfig | None = None,
) -> list[tuple[int, int]]:
    """Return [start_sample, end_sample) spans of detected utterances."""
    cfg = cfg or VADConfig()
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    frame = max(1, int(sr * cfg.frame_ms / 1000))
    n_frames = len(pcm) // frame
    tail_frames = max(1, math.ceil(cfg.tail_silence_ms / cfg.frame_ms))
    max_frames = max(1, int(cfg.max_utt_ms / cfg.frame_ms))
    spans: list[tuple[int, int]] = []
    active = False
    start_frame = 0
    silence_frames = 0
    pending = 0

    for i in range(n_frames):
        voiced = _rms(pcm[i * frame : (i + 1) * frame]) > cfg.energy_threshold
        if not active:
            if voiced:
                pending += 1
                if pending >= cfg.start_frames:
                    active = True
                    start_frame = i - cfg.start_frames + 1
                    silence_frames = 0
            else:
                pending = 0
            continue
        if voiced:
            silence_frames = 0
        else:
            silence_frames += 1
            if silence_frames >= tail_frames:
                end_frame = i - silence_frames + 1
                spans.append((start_frame * frame, max(start_frame * frame + 1, end_frame * frame)))
                active = False
                silence_frames = 0
                pending = 0
                continue
        if (i - start_frame + 1) >= max_frames:
            end_frame = i + 1
            spans.append((start_frame * frame, end_frame * frame))
            start_frame = end_frame
            silence_frames = 0
    if active:
        spans.append((start_frame * frame, len(pcm)))
    return [
        (a, b)
        for a, b in spans
        if (b - a) / float(sr) * 1000.0 >= cfg.min_speech_ms
    ]


def split_utterances(
    pcm: np.ndarray,
    sr: int = 16000,
    cfg: VADConfig | None = None,
    *,
    source: str = "",
) -> list[Utterance]:
    out: list[Utterance] = []
    for a, b in energy_vad(pcm, sr, cfg):
        out.append(
            Utterance(
                request_id=new_request_id(),
                pcm=np.asarray(pcm[a:b], dtype=np.float32).copy(),
                sr=sr,
                start_ms=a / float(sr) * 1000.0,
                end_ms=b / float(sr) * 1000.0,
                source=source,
            )
        )
    return out


class VADStream:
    """Incremental frame feeder. Re-runs endpointing on the small pending buffer.

    O(n^2) over one utterance; fine for <=15 s prototypes. A real deployment
    would replace this with a stateful streaming encoder (S1 chunk-causal).
    """

    def __init__(self, sr: int = 16000, cfg: VADConfig | None = None, source: str = "") -> None:
        self.sr = sr
        self.cfg = cfg or VADConfig()
        self.source = source
        self.buf = np.zeros(0, dtype=np.float32)
        self._consumed_ms = 0.0
        self._barge = False

    def feed(self, chunk: np.ndarray) -> list[Utterance]:
        self.buf = np.concatenate([self.buf, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        spans = energy_vad(self.buf, self.sr, self.cfg)
        out: list[Utterance] = []
        last_end = 0
        for a, b in spans:
            out.append(
                Utterance(
                    request_id=new_request_id(),
                    pcm=self.buf[a:b].copy(),
                    sr=self.sr,
                    start_ms=self._consumed_ms + a / float(self.sr) * 1000.0,
                    end_ms=self._consumed_ms + b / float(self.sr) * 1000.0,
                    source=self.source,
                    barge_in=self._barge,
                )
            )
            self._barge = False
            last_end = b
        if last_end:
            self._consumed_ms += last_end / float(self.sr) * 1000.0
            self.buf = self.buf[last_end:]
        return out

    def barge_in(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self._barge = True

    def flush(self) -> list[Utterance]:
        if self.buf.size < self.cfg.min_speech_ms * self.sr / 1000.0:
            self.buf = np.zeros(0, dtype=np.float32)
            return []
        utt = Utterance(
            request_id=new_request_id(),
            pcm=self.buf.copy(),
            sr=self.sr,
            start_ms=self._consumed_ms,
            end_ms=self._consumed_ms + self.buf.size / float(self.sr) * 1000.0,
            source=self.source,
            barge_in=self._barge,
        )
        self._barge = False
        self.buf = np.zeros(0, dtype=np.float32)
        return [utt]
