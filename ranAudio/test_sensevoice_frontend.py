#!/usr/bin/env python3
"""SenseVoice frontend LFR shape (no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.sensevoice_apm import INPUT_SIZE, wav_to_lfr


def test_lfr_dim() -> None:
    wav = np.zeros(16000, dtype=np.float32)
    feat = wav_to_lfr(wav)
    assert feat.ndim == 2
    assert feat.shape[-1] == INPUT_SIZE == 560
    assert feat.shape[0] >= 1


if __name__ == "__main__":
    test_lfr_dim()
    print("ok")
