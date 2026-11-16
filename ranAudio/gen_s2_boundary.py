#!/usr/bin/env python3
"""Write the boundary TTS jobs, or merge finished wavs into a mix jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import DATA_DIR, S2_AUDIO_JSONL, rel_wav
from ranAudio.protocol import format_fc_base, format_nofc
from ranAudio.s2_catalog import boundary_jobs, jobs_summary, read_jsonl, write_jsonl

JOBS = DATA_DIR / "s2_boundary_jobs.jsonl"
WAVS = DATA_DIR / "s2_boundary_wavs"
MERGED = DATA_DIR / "s2_audio_boundary.jsonl"


def _wav_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def assemble(jobs_path: Path, wav_dir: Path, base_path: Path, out_path: Path) -> int:
    rows = list(read_jsonl(base_path))
    seen = {str(r.get("id") or "") for r in rows}
    added = 0
    missing = 0
    for row in read_jsonl(jobs_path):
        sid = str(row.get("id") or "")
        wav = wav_dir / f"{sid}.wav"
        if not _wav_ok(wav):
            missing += 1
            continue
        if sid in seen:
            continue
        user = str(row.get("user") or "").strip()
        rec = {
            "id": sid,
            "wav": rel_wav(wav),
            "user": user,
            "source": "boundary",
            "engine": row.get("engine"),
            "voice": row.get("voice"),
        }
        bucket = str(row.get("bucket") or "")
        if bucket == "fc":
            rec["target"] = format_fc_base(user)
            rec["route"] = "fc"
            rec["venue"] = "base"
        elif bucket == "asr":
            body = str(row.get("body") or "").strip()
            rec["target"] = format_nofc(body)
            rec["route"] = "nofc"
            rec["kind"] = "asr"
            rec["body"] = body
            rec["trap"] = bool(row.get("trap"))
        else:
            reply = str(row.get("reply") or "").strip()
            rec["target"] = format_nofc(reply)
            rec["route"] = "nofc"
            rec["kind"] = "chat"
            rec["reply"] = reply
        rows.append(rec)
        seen.add(sid)
        added += 1
    n = write_jsonl(out_path, rows)
    print(
        f"assembled base={len(rows) - added} added={added} missing={missing} -> {out_path} n={n}",
        flush=True,
    )
    if missing:
        print(f"warn missing wavs={missing}; continuing with the wavs that exist", flush=True)
    return 0 if added else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(JOBS))
    ap.add_argument("--wav-dir", default=str(WAVS))
    ap.add_argument("--base", default=str(S2_AUDIO_JSONL))
    ap.add_argument("--out", default=str(MERGED))
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()

    if args.assemble:
        return assemble(Path(args.jobs), Path(args.wav_dir), Path(args.base), Path(args.out))

    jobs = boundary_jobs()
    out = Path(args.jobs)
    n = write_jsonl(out, jobs)
    summary = jobs_summary(jobs)
    out.with_suffix(".meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {n} boundary jobs -> {out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
