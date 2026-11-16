#!/usr/bin/env python3
"""Extract wav clips from WenetSpeech using Tier-A JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def cut_one(audio_root: Path, row: dict, out_path: Path) -> None:
    src = audio_root / row["audio_path"]
    if not src.is_file():
        raise FileNotFoundError(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{float(row['start']):.4f}",
        "-t",
        f"{float(row['duration']):.4f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="WenetSpeech DATA_UNTAR_DIRECTORY",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max", type=int, default=0, help="Max utterances (0=all)")
    args = ap.parse_args()

    n = 0
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row["utt_id"]
            out = args.out_dir / f"{uid}.wav"
            try:
                cut_one(args.audio_root, row, out)
                n += 1
                if n % 50 == 0:
                    print(f"extracted {n}", flush=True)
            except Exception as exc:
                print(f"[skip] {uid}: {exc}", flush=True)
            if args.max > 0 and n >= args.max:
                break
    print(f"done: {n} wavs -> {args.out_dir}")


if __name__ == "__main__":
    main()
