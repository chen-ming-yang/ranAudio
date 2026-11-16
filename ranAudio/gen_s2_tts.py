#!/usr/bin/env python3
"""Synthesize S2 TTS jobs to 16 kHz mono wavs. One --engine per process.

Engines: edge | qwen_custom | qwen_design | qwen_clone | index_clone
Skip existing non-empty wavs. Failures go to tmp/s2_tts_fail.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import (
    INDEXTTS_CFG,
    INDEXTTS_DIR,
    QWEN_BASE,
    QWEN_CUSTOM,
    QWEN_DESIGN,
    S2_TTS_JOBS,
    S2_TTS_WAVS,
    TMP_DIR,
    rel_wav,
)
from ranAudio.s2_catalog import read_jsonl

ENGINES = ("edge", "qwen_custom", "qwen_design", "qwen_clone", "index_clone")


def _to_mono(wav: np.ndarray) -> np.ndarray:
    arr = np.asarray(wav)
    if arr.ndim == 2:
        arr = arr.mean(axis=1) if arr.shape[1] <= 8 else arr.mean(axis=0)
    return arr.astype(np.float32, copy=False)


def _to_float(wav: np.ndarray) -> np.ndarray:
    arr = np.asarray(wav)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.float32(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32, copy=False)
    return arr


def resample_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = _to_mono(_to_float(wav))
    if sr == 16000 or wav.size == 0:
        return wav
    n = max(1, int(round(wav.size * 16000 / float(sr))))
    x_old = np.linspace(0.0, 1.0, wav.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, wav.astype(np.float64)).astype(np.float32)


def write_wav_16k(wav: np.ndarray, sr: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio = resample_16k(wav, sr)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        audio = audio / peak * 0.95
    sf.write(str(dst), audio, 16000, subtype="PCM_16")


def ffmpeg_to_16k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wav_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _append_fail(row: dict, err: str) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"id": row.get("id"), "engine": row.get("engine"), "error": err}
    with (TMP_DIR / "s2_tts_fail.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def _edge_one(text: str, voice: str, mp3: Path) -> None:
    import edge_tts

    last: Exception | None = None
    for attempt in range(4):
        try:
            await asyncio.wait_for(
                edge_tts.Communicate(text, voice=voice).save(str(mp3)),
                timeout=45,
            )
            if mp3.is_file() and mp3.stat().st_size > 0:
                return
            last = RuntimeError(f"empty mp3 {mp3}")
        except Exception as exc:  # noqa: BLE001
            last = exc
            await asyncio.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"edge-tts failed text={text!r}: {last}") from last


async def run_edge(jobs: list[dict], wav_dir: Path, concurrency: int) -> tuple[int, int]:
    sem = asyncio.Semaphore(max(1, concurrency))
    ok = skip = 0

    async def one(row: dict) -> None:
        nonlocal ok, skip
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        if wav_ok(dst):
            skip += 1
            return
        text = str(row.get("user") or "").strip()
        voice = str(row.get("voice") or "zh-CN-XiaoxiaoNeural")
        async with sem:
            print(f"edge {sid} {voice} {text[:24]!r}", flush=True)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    mp3 = Path(tmp) / f"{sid}.mp3"
                    await _edge_one(text, voice, mp3)
                    await asyncio.to_thread(ffmpeg_to_16k, mp3, dst)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                _append_fail(row, str(exc))
                print(f"fail {sid}: {exc}", flush=True)

    await asyncio.gather(*[one(r) for r in jobs])
    return ok, skip


def _load_qwen(model_dir: Path, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    kwargs = dict(device_map=device, dtype=torch.bfloat16)
    try:
        return Qwen3TTSModel.from_pretrained(str(model_dir), attn_implementation="sdpa", **kwargs)
    except Exception:
        return Qwen3TTSModel.from_pretrained(str(model_dir), **kwargs)


def _chunked(items: list, n: int):
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def run_qwen_custom(
    jobs: list[dict], wav_dir: Path, device: str, batch_size: int = 8
) -> tuple[int, int]:
    """Batched CustomVoice TTS. API accepts List[str] for text/speaker."""
    tts = _load_qwen(QWEN_CUSTOM, device)
    ok = skip = 0
    pending: list[dict] = []
    for row in jobs:
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        if wav_ok(dst):
            skip += 1
        else:
            pending.append(row)

    total = len(jobs)
    done_base = skip
    bs = max(1, int(batch_size))
    print(f"custom batch_size={bs} pending={len(pending)} skip={skip}", flush=True)

    def _one(row: dict) -> bool:
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        text = str(row.get("user") or "").strip()
        speaker = str(row.get("voice") or "Vivian")
        try:
            wavs, sr = tts.generate_custom_voice(
                text=text, language="Chinese", speaker=speaker
            )
            write_wav_16k(wavs[0], int(sr), dst)
            return True
        except Exception as exc:  # noqa: BLE001
            _append_fail(row, str(exc))
            print(f"fail {sid}: {exc}", flush=True)
            return False

    for bi, batch in enumerate(_chunked(pending, bs), 1):
        t0 = time.time()
        texts = [str(r.get("user") or "").strip() for r in batch]
        speakers = [str(r.get("voice") or "Vivian") for r in batch]
        langs = ["Chinese"] * len(batch)
        try:
            wavs, sr = tts.generate_custom_voice(
                text=texts, speaker=speakers, language=langs
            )
            if len(wavs) != len(batch):
                raise RuntimeError(f"batch size mismatch got={len(wavs)} want={len(batch)}")
            for row, wav in zip(batch, wavs):
                write_wav_16k(wav, int(sr), wav_dir / f"{row['id']}.wav")
                ok += 1
            print(
                f"custom batch[{bi}] n={len(batch)} "
                f"progress={done_base + ok}/{total} {time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"custom batch[{bi}] fail -> fallback one-by-one: {exc}", flush=True)
            for row in batch:
                if _one(row):
                    ok += 1
    return ok, skip


def run_qwen_design(
    jobs: list[dict], wav_dir: Path, device: str, batch_size: int = 8
) -> tuple[int, int]:
    """Batched VoiceDesign TTS."""
    tts = _load_qwen(QWEN_DESIGN, device)
    ok = skip = 0
    pending: list[dict] = []
    for row in jobs:
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        if wav_ok(dst):
            skip += 1
        else:
            pending.append(row)

    total = len(jobs)
    done_base = skip
    bs = max(1, int(batch_size))
    print(f"design batch_size={bs} pending={len(pending)} skip={skip}", flush=True)

    def _one(row: dict) -> bool:
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        text = str(row.get("user") or "").strip()
        instruct = str(row.get("instruct") or "成年男声，平稳口语，语速中等")
        try:
            wavs, sr = tts.generate_voice_design(
                text=text, language="Chinese", instruct=instruct
            )
            write_wav_16k(wavs[0], int(sr), dst)
            return True
        except Exception as exc:  # noqa: BLE001
            _append_fail(row, str(exc))
            print(f"fail {sid}: {exc}", flush=True)
            return False

    for bi, batch in enumerate(_chunked(pending, bs), 1):
        t0 = time.time()
        texts = [str(r.get("user") or "").strip() for r in batch]
        instructs = [
            str(r.get("instruct") or "成年男声，平稳口语，语速中等") for r in batch
        ]
        langs = ["Chinese"] * len(batch)
        try:
            wavs, sr = tts.generate_voice_design(
                text=texts, instruct=instructs, language=langs
            )
            if len(wavs) != len(batch):
                raise RuntimeError(f"batch size mismatch got={len(wavs)} want={len(batch)}")
            for row, wav in zip(batch, wavs):
                write_wav_16k(wav, int(sr), wav_dir / f"{row['id']}.wav")
                ok += 1
            print(
                f"design batch[{bi}] n={len(batch)} "
                f"progress={done_base + ok}/{total} {time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"design batch[{bi}] fail -> fallback one-by-one: {exc}", flush=True)
            for row in batch:
                if _one(row):
                    ok += 1
    return ok, skip


def run_qwen_clone(jobs: list[dict], wav_dir: Path, device: str) -> tuple[int, int]:
    tts = _load_qwen(QWEN_BASE, device)
    ok = skip = 0
    for i, row in enumerate(jobs, 1):
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        if wav_ok(dst):
            skip += 1
            continue
        text = str(row.get("user") or "").strip()
        ref_wav = str(row.get("ref_wav") or "")
        ref_text = str(row.get("ref_text") or "")
        t0 = time.time()
        try:
            wavs, sr = tts.generate_voice_clone(
                text=text, language="Chinese", ref_audio=ref_wav, ref_text=ref_text
            )
            write_wav_16k(wavs[0], int(sr), dst)
            ok += 1
            print(f"qclone [{i}/{len(jobs)}] {sid} {time.time()-t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            _append_fail(row, str(exc))
            print(f"fail {sid}: {exc}", flush=True)
    return ok, skip


def run_index_clone(jobs: list[dict], wav_dir: Path, device: str) -> tuple[int, int]:
    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=str(INDEXTTS_CFG),
        model_dir=str(INDEXTTS_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_torch_compile=False,
        device=device if device.startswith("cuda") else "cuda:0",
    )
    ok = skip = 0
    for i, row in enumerate(jobs, 1):
        sid = str(row["id"])
        dst = wav_dir / f"{sid}.wav"
        if wav_ok(dst):
            skip += 1
            continue
        text = str(row.get("user") or "").strip()
        ref_wav = str(row.get("ref_wav") or "")
        t0 = time.time()
        try:
            out = tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=None, verbose=False)
            if out is None:
                raise RuntimeError("IndexTTS returned None")
            sr, wav = out
            write_wav_16k(np.asarray(wav), int(sr), dst)
            ok += 1
            print(f"index [{i}/{len(jobs)}] {sid} {time.time()-t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            _append_fail(row, str(exc))
            print(f"fail {sid}: {exc}", flush=True)
    return ok, skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(S2_TTS_JOBS))
    ap.add_argument("--wav-dir", default=str(S2_TTS_WAVS))
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--device", default=os.environ.get("S2_TTS_DEVICE", "cuda:0"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("S2_TTS_BATCH", "8")),
        help="GPU TTS micro-batch (qwen_custom / qwen_design)",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    jobs = [r for r in read_jsonl(Path(args.jobs)) if r.get("engine") == args.engine]
    if args.limit:
        jobs = jobs[: args.limit]
    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    print(f"engine={args.engine} n={len(jobs)} -> {wav_dir}", flush=True)
    if not jobs:
        return 0

    t0 = time.time()
    if args.engine == "edge":
        ok, skip = asyncio.run(run_edge(jobs, wav_dir, args.concurrency))
    elif args.engine == "qwen_custom":
        ok, skip = run_qwen_custom(jobs, wav_dir, args.device, batch_size=args.batch_size)
    elif args.engine == "qwen_design":
        ok, skip = run_qwen_design(jobs, wav_dir, args.device, batch_size=args.batch_size)
    elif args.engine == "qwen_clone":
        ok, skip = run_qwen_clone(jobs, wav_dir, args.device)
    else:
        ok, skip = run_index_clone(jobs, wav_dir, args.device)

    n = len(jobs)
    print(
        f"done engine={args.engine} ok={ok} skip={skip} fail={n - ok - skip} "
        f"t={time.time()-t0:.0f}s",
        flush=True,
    )
    made = ok + skip
    if n and made / n < 0.8:
        print(f"too many failures {made}/{n}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
