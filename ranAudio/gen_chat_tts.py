#!/usr/bin/env python3
"""Synthesize chat negative wavs via edge-tts (16 kHz mono)."""

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

from ranAudio.paths import CHAT_TEXTS, CHAT_WAVS

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


async def _synthesize_one(text: str, voice: str, mp3_path: Path, retries: int = 4) -> None:
    import edge_tts

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            await edge_tts.Communicate(text, voice=voice).save(str(mp3_path))
            if mp3_path.is_file() and mp3_path.stat().st_size > 0:
                return
            last_exc = RuntimeError(f"empty mp3 {mp3_path}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            await asyncio.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"TTS failed text={text!r}: {last_exc}") from last_exc


def _to_wav_16k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _one(
    i: int,
    n: int,
    row: dict,
    tmp_dir: Path,
    out_dir: Path,
    voice: str,
    sem: asyncio.Semaphore,
) -> None:
    sid = str(row.get("id") or f"chat_{i:04d}")
    text = str(row.get("text") or "").strip()
    if not text:
        return
    wav_path = out_dir / f"{sid}.wav"
    if wav_path.is_file() and wav_path.stat().st_size > 0:
        return
    async with sem:
        print(f"[{i + 1}/{n}] TTS {text!r}", flush=True)
        mp3_path = tmp_dir / f"{sid}.mp3"
        await _synthesize_one(text, voice=voice, mp3_path=mp3_path)
        await asyncio.to_thread(_to_wav_16k, mp3_path, wav_path)


async def _run(rows: list[dict], out_dir: Path, voice: str, concurrency: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tasks = [_one(i, len(rows), row, tmp_dir, out_dir, voice, sem) for i, row in enumerate(rows)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
        for i, err in failed:
            print(f"fail {rows[i].get('id')}: {err}", flush=True)
        if failed:
            raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(CHAT_TEXTS))
    ap.add_argument("--out-dir", default=str(CHAT_WAVS))
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with Path(args.texts).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_samples:
        rows = rows[: args.max_samples]
    asyncio.run(_run(rows, Path(args.out_dir), args.voice, args.concurrency))
    print(f"done n={len(rows)} -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
