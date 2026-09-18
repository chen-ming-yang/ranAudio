#!/usr/bin/env python3
"""Build ASR + FC jsonl from existing car wavs / texts.jsonl (v1.0 baseline).

For v1.1 judge data (need_fc + chat negatives), use build_judge_samples.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from finetune_fc.fc_format import extract_slots_from_text, load_tools
from ranAudio.paths import CAR_TEXTS, CAR_WAVS, DATA_DIR, rel_wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(CAR_TEXTS))
    ap.add_argument("--wav-dir", default=str(CAR_WAVS))
    ap.add_argument("--out", default=str(DATA_DIR / "car_audio.jsonl"))
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tools = load_tools()
    n = 0
    missing = 0
    with Path(args.texts).open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id") or "")
            wav = wav_dir / f"{sid}.wav"
            if not wav.is_file():
                missing += 1
                continue
            rel = rel_wav(wav)
            text = str(row.get("text") or "")
            tool = str(row.get("tool") or "")
            rec = {
                "id": sid,
                "wav": rel,
                "text": text,
                "tool": tool,
                "arguments": extract_slots_from_text(tool, text),
                "tools": tools,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {out} missing_wav={missing}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
