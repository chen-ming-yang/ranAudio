"""ranAudio paths. Project is /cmy/cmy/ranAudio, sibling of MiniCPM-o-Demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RANAUDIO = PACKAGE_DIR.parent
DEMO_ROOT = Path(os.environ.get("MINICPMO_DEMO", str(RANAUDIO.parent / "MiniCPM-o-Demo"))).resolve()
# Back-compat alias: Demo tree (models / finetune_fc / slave)
REPO_ROOT = DEMO_ROOT

MINICPM5 = DEMO_ROOT / "models" / "MiniCPM5-2B"
MINICPMO = DEMO_ROOT / "models" / "MiniCPM-o-4_5"
SENSEVOICE = Path(os.environ.get("SENSEVOICE", str(DEMO_ROOT / "models" / "SenseVoiceSmall"))).resolve()
CAR_TEXTS = DEMO_ROOT / "finetune_fc" / "demo_car_fc" / "texts.jsonl"
CAR_WAVS = DEMO_ROOT / "finetune_fc" / "demo_car_fc" / "wavs"
DATA_DIR = RANAUDIO / "data"
RUNS_DIR = RANAUDIO / "runs"
TMP_DIR = RANAUDIO / "tmp"
CHAT_TEXTS = DATA_DIR / "chat_texts.jsonl"
CHAT_WAVS = DATA_DIR / "chat_wavs"
JUDGE_JSONL = DATA_DIR / "judge_audio.jsonl"
CAR_AUDIO_JSONL = DATA_DIR / "car_audio.jsonl"
ROUTE_JSONL = DATA_DIR / "route_audio.jsonl"
WENET_S_JSONL = DATA_DIR / "wenet_s.jsonl"
REAL_ASR = Path(os.environ.get("REAL_ASR", str(RANAUDIO.parent / "real-asr"))).resolve()
WENET_MANIFEST = REAL_ASR / "data" / "manifests" / "wenetspeech"
AISHELL_MANIFEST = REAL_ASR / "data" / "manifests" / "aishell"
WENET_FLAC = Path(os.environ.get("WENET_FLAC", "/cmy/cmy/WenetSpeech/flac_16k")).resolve()


def bootstrap_sys_path() -> None:
    """`import ranAudio` from this project; `import finetune_fc` from Demo."""
    project = str(RANAUDIO)
    demo = str(DEMO_ROOT)
    for p in (project, demo):
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_wav(rel: str | Path) -> Path:
    """Resolve jsonl wav paths against this project and Demo."""
    p = Path(rel)
    if p.is_absolute():
        return p
    s = str(rel).lstrip("./")
    if s.startswith("ranAudio/"):
        s = s[len("ranAudio/") :]
    for root in (RANAUDIO, DEMO_ROOT):
        cand = (root / s).resolve()
        if cand.is_file():
            return cand
    if s.startswith("finetune_fc/") or s.startswith("models/"):
        return (DEMO_ROOT / s).resolve()
    return (RANAUDIO / s).resolve()


def rel_wav(wav: Path) -> str:
    wav = wav.resolve()
    try:
        return str(wav.relative_to(DEMO_ROOT.resolve()))
    except ValueError:
        pass
    try:
        return str(wav.relative_to(RANAUDIO.resolve()))
    except ValueError:
        return str(wav)
