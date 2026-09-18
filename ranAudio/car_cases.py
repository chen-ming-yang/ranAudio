"""Shared 22-tool car eval ids + chat negatives."""

from __future__ import annotations

import json
from pathlib import Path

from ranAudio.paths import CAR_TEXTS, CAR_WAVS, CHAT_WAVS, JUDGE_JSONL

TEST_TOOLS = [
    "enable_bluetooth",
    "disable_bluetooth",
    "navigate_to",
    "cancel_navigation",
    "enable_ac",
    "disable_ac",
    "set_temperature",
    "volume_up",
    "volume_down",
    "set_volume",
    "open_window",
    "close_window",
    "answer_call",
    "reject_call",
    "play_music",
    "pause_music",
    "next_track",
    "previous_track",
    "enable_seat_heating",
    "disable_seat_heating",
    "enable_headlights",
    "disable_headlights",
]


def load_judge_by_id(path: Path = JUDGE_JSONL) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.is_file():
        return by_id
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[str(row.get("id") or "")] = row
    return by_id


def load_car_texts() -> dict[str, dict]:
    texts: dict[str, dict] = {}
    if not CAR_TEXTS.is_file():
        return texts
    with CAR_TEXTS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts[str(row.get("id") or "")] = row
    return texts


def car_wav_cases(by_id: dict[str, dict] | None = None) -> list[dict]:
    by_id = by_id if by_id is not None else load_judge_by_id()
    texts = load_car_texts()
    out: list[dict] = []
    for name in TEST_TOOLS:
        wav = CAR_WAVS / f"{name}_000.wav"
        if not wav.is_file():
            continue
        row = by_id.get(wav.stem) or texts.get(wav.stem) or {}
        asr = str(row.get("asr") or row.get("text") or "")
        out.append(
            {
                "id": wav.stem,
                "wav": wav,
                "asr": asr,
                "text": asr,
                "need_fc": True,
                "tool": name,
            }
        )
    return out


def chat_wav_cases(n: int = 22, by_id: dict[str, dict] | None = None) -> list[dict]:
    by_id = by_id if by_id is not None else load_judge_by_id()
    ids = sorted(p.stem for p in CHAT_WAVS.glob("chat_*.wav"))[: max(0, n)]
    out: list[dict] = []
    for sid in ids:
        wav = CHAT_WAVS / f"{sid}.wav"
        row = by_id.get(sid) or {}
        asr = str(row.get("asr") or row.get("text") or "")
        out.append(
            {
                "id": sid,
                "wav": wav,
                "asr": asr,
                "text": asr,
                "need_fc": False,
                "reply": str(row.get("reply") or ""),
                "tool": "",
            }
        )
    return out
