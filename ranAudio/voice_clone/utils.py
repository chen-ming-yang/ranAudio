"""构音障碍音色克隆 — 工具函数。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 音频工具 ──────────────────────────────────────────

def load_audio(path: str | Path, sr: int | None = None):
    """加载音频，返回 (waveform, sample_rate)。"""
    import numpy as np
    import soundfile as sf

    wav, orig_sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr is not None and orig_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
        return wav, sr
    return wav, orig_sr


def save_audio(path: str | Path, wav, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)


def audio_duration(path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.duration


# ── JSONL 工具 ────────────────────────────────────────

def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, items: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── 文本工具 ──────────────────────────────────────────

def clean_text(text: str) -> str:
    """清洗文本：去除多余空白。"""
    return text.strip()
