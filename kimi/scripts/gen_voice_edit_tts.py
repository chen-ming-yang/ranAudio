#!/usr/bin/env python3
"""Synthesize voice-edit instruction wavs with multiple TTS engines.

Reuses ranAudio.gen_s2_tts (edge / qwen_custom / qwen_design / index_clone).
Writes 16 kHz mono wavs to eval_voice_edit/wavs/{engine}/{id}.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

KIMI = Path(__file__).resolve().parents[1]
ROOT = KIMI.parent
EVAL = KIMI / "eval_voice_edit"
CASES = EVAL / "cases.jsonl"
WAVS = EVAL / "wavs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ranAudio._boot  # noqa: F401
from ranAudio import gen_s2_tts as tts  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_job(row: dict) -> dict:
    job = {
        "id": row["id"],
        "user": row["instruction"],
        "engine": row.get("engine") or "edge",
        "voice": row.get("voice") or "zh-CN-XiaoxiaoNeural",
    }
    if row.get("instruct"):
        job["instruct"] = row["instruct"]
    return job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--wav-root", default=str(WAVS))
    ap.add_argument(
        "--engines",
        default="edge",
        help="comma list: edge,qwen_custom,qwen_design,index_clone",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="resynthesize even if wav exists")
    args = ap.parse_args()

    want = {e.strip() for e in args.engines.split(",") if e.strip()}
    rows = [r for r in load_cases(Path(args.cases)) if (r.get("engine") or "edge") in want]
    wav_root = Path(args.wav_root)
    by_engine: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        eng = r.get("engine") or "edge"
        dst_dir = wav_root / eng
        dst = dst_dir / f"{r['id']}.wav"
        if args.force and dst.is_file():
            dst.unlink()
        by_engine[eng].append(to_job(r))

    total_ok = total_skip = 0
    for eng, jobs in by_engine.items():
        out_dir = wav_root / eng
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== engine={eng} n={len(jobs)} -> {out_dir}", flush=True)
        try:
            if eng == "edge":
                ok, skip = asyncio.run(tts.run_edge(jobs, out_dir, args.concurrency))
            elif eng == "qwen_custom":
                ok, skip = tts.run_qwen_custom(jobs, out_dir, args.device)
            elif eng == "qwen_design":
                ok, skip = tts.run_qwen_design(jobs, out_dir, args.device)
            elif eng == "qwen_clone":
                ok, skip = tts.run_qwen_clone(jobs, out_dir, args.device)
            elif eng == "index_clone":
                ok, skip = tts.run_index_clone(jobs, out_dir, args.device)
            else:
                print(f"skip unknown engine {eng}", flush=True)
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"engine {eng} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"engine={eng} ok={ok} skip={skip}", flush=True)
        total_ok += ok
        total_skip += skip

    print(f"DONE ok={total_ok} skip={total_skip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
