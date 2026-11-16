#!/usr/bin/env python3
"""Tiny S2 preview: a few FC / chat / asr rows with edge-tts 16 kHz wavs."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.build_judge_samples import default_chat_reply
from ranAudio.paths import DATA_DIR, rel_wav
from ranAudio.protocol import format_fc_base, format_nofc

PREVIEW = [
    {
        "id": "preview_fc_bluetooth",
        "route": "fc",
        "kind": None,
        "venue": "base",
        "user": "打开蓝牙",
        "voice": "zh-CN-XiaoxiaoNeural",
    },
    {
        "id": "preview_fc_nav",
        "route": "fc",
        "kind": None,
        "venue": "base",
        "user": "导航去公司",
        "voice": "zh-CN-YunxiNeural",
    },
    {
        "id": "preview_fc_ac",
        "route": "fc",
        "kind": None,
        "venue": "base",
        "user": "把空调调到二十六度",
        "voice": "zh-CN-XiaoyiNeural",
    },
    {
        "id": "preview_fc_weather",
        "route": "fc",
        "kind": None,
        "venue": "base",
        "user": "今天天气怎么样",
        "voice": "zh-CN-YunyangNeural",
    },
    {
        "id": "preview_chat_who",
        "route": "nofc",
        "kind": "chat",
        "venue": None,
        "user": "你是谁",
        "voice": "zh-CN-liaoning-XiaobeiNeural",
    },
    {
        "id": "preview_chat_joke",
        "route": "nofc",
        "kind": "chat",
        "venue": None,
        "user": "讲个笑话",
        "voice": "zh-CN-YunjianNeural",
    },
    {
        "id": "preview_asr_zhuanxie",
        "route": "nofc",
        "kind": "asr",
        "venue": None,
        "user": "请转写：今天会议取消了",
        "body": "今天会议取消了",
        "voice": "zh-CN-XiaoxiaoNeural",
    },
    {
        "id": "preview_asr_xiecheng",
        "route": "nofc",
        "kind": "asr",
        "venue": None,
        "user": "写成文字，明天三点开会",
        "body": "明天三点开会",
        "voice": "zh-CN-YunxiNeural",
    },
    {
        "id": "preview_asr_fanyi",
        "route": "nofc",
        "kind": "asr",
        "venue": None,
        "user": "翻译成文字：周报发到群里",
        "body": "周报发到群里",
        "voice": "zh-CN-XiaoyiNeural",
    },
    {
        "id": "preview_asr_trap",
        "route": "nofc",
        "kind": "asr",
        "venue": None,
        "user": "翻译成文字：打开蓝牙",
        "body": "打开蓝牙",
        "voice": "zh-CN-YunyangNeural",
    },
]


def _target(row: dict) -> str:
    if row["route"] == "fc":
        return format_fc_base(row["user"])
    if row["kind"] == "asr":
        return format_nofc(row["body"])
    return format_nofc(default_chat_reply(row["user"]))


async def _synth(text: str, voice: str, mp3: Path) -> None:
    import edge_tts

    last: Exception | None = None
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, voice=voice).save(str(mp3))
            if mp3.is_file() and mp3.stat().st_size > 0:
                return
            last = RuntimeError(f"empty mp3 {mp3}")
        except Exception as exc:  # noqa: BLE001
            last = exc
            await asyncio.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"TTS failed text={text!r}: {last}") from last


def _to_wav_16k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _one(row: dict, wav_dir: Path, tmp: Path) -> dict:
    wav = wav_dir / f"{row['id']}.wav"
    mp3 = tmp / f"{row['id']}.mp3"
    print(f"TTS {row['id']} {row['user']!r} {row['voice']}", flush=True)
    await _synth(row["user"], row["voice"], mp3)
    await asyncio.to_thread(_to_wav_16k, mp3, wav)
    rec = {
        "id": row["id"],
        "wav": rel_wav(wav),
        "user": row["user"],
        "target": _target(row),
        "route": row["route"],
        "engine": "edge-tts",
        "voice": row["voice"],
    }
    if row.get("kind"):
        rec["kind"] = row["kind"]
    if row.get("venue"):
        rec["venue"] = row["venue"]
    return rec


async def _run(out_jsonl: Path, wav_dir: Path) -> None:
    wav_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        rows = await asyncio.gather(*[_one(r, wav_dir, Path(tmp)) for r in PREVIEW])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out_jsonl}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "s2_preview.jsonl"))
    ap.add_argument("--wav-dir", default=str(DATA_DIR / "s2_preview_wavs"))
    args = ap.parse_args()
    asyncio.run(_run(Path(args.out), Path(args.wav_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
