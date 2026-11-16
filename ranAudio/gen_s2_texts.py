#!/usr/bin/env python3
"""Emit S2 TTS jobs jsonl (no wavs). Deterministic seed; safe to rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import S2_TTS_JOBS
from ranAudio.s2_catalog import (
    N_ASR_TTS,
    N_CHAT_TTS,
    N_FC_TTS,
    SEED,
    build_jobs,
    jobs_summary,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S2_TTS_JOBS))
    ap.add_argument("--n-fc", type=int, default=N_FC_TTS)
    ap.add_argument("--n-chat", type=int, default=N_CHAT_TTS)
    ap.add_argument("--n-asr", type=int, default=N_ASR_TTS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    jobs = build_jobs(n_fc=args.n_fc, n_chat=args.n_chat, n_asr=args.n_asr, seed=args.seed)
    out = Path(args.out)
    n = write_jsonl(out, jobs)
    summary = jobs_summary(jobs)
    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {n} jobs -> {out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
