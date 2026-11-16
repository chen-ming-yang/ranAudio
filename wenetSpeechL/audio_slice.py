#!/usr/bin/env python3
"""Load a time slice from a long WenetSpeech opus/wav as 16 kHz mono float32."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

TARGET_SR = 16000


def load_slice(
    audio_path: str,
    start: float,
    duration: float,
    target_sr: int = TARGET_SR,
) -> Tuple[np.ndarray, int]:
    """Return (waveform float32 mono, sample_rate). Prefer ffmpeg for opus seek."""
    if duration <= 0:
        raise ValueError(f"invalid duration={duration}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    # ffmpeg: accurate seek for opus
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start:.4f}",
        "-t",
        f"{duration:.4f}",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found; required to slice opus") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg failed for {audio_path} [{start},{duration}]: {err}") from exc

    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise RuntimeError(f"empty audio slice: {audio_path} start={start} dur={duration}")
    return wav, target_sr


def write_temp_wav(
    audio_path: str,
    start: float,
    duration: float,
    out_path: Optional[str] = None,
    target_sr: int = TARGET_SR,
) -> str:
    """Write a 16-bit PCM wav slice; return path. Caller should delete when done."""
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="wslic_")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.4f}",
        "-t",
        f"{duration:.4f}",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-acodec",
        "pcm_s16le",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg wav write failed: {err}") from exc
    return out_path


def cer_chars(ref: str, hyp: str) -> Tuple[float, int, int]:
    """Character CER with punctuation stripped. Returns (cer, errors, ref_len)."""
    ref_n = strip_punctuation(ref)
    hyp_n = strip_punctuation(hyp)
    if not ref_n:
        return (0.0 if not hyp_n else 1.0), len(hyp_n), 0
    errs = edit_distance(ref_n, hyp_n)
    return errs / len(ref_n), errs, len(ref_n)


PUNCTUATION = set(
    "。，、；：？！“”‘’【】《》（）(){}[],.;:?!'\""
    "\n\r\t·…—─"
)


def strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCTUATION and not ch.isspace())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # banded DP for typical short Chinese utts
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]
