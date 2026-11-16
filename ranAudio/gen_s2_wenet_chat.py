#!/usr/bin/env python3
"""Screen Wenet questions, MiniCPM5 T2T replies, write S2 NOFC chat wavs.

Does not modify data/wenet_s.jsonl. Resume-safe: existing out jsonl is kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import (
    MINICPM5,
    MINICPMO,
    S2_WENET_CHAT,
    S2_WENET_WAVS,
    SENSEVOICE,
    WENET_S_JSONL,
    rel_wav,
)
from ranAudio.protocol import format_nofc
from ranAudio.s2_catalog import read_jsonl
from ranAudio.s2_wenet import T2T_SYSTEM, collect_candidates, ok_reply
from ranAudio.wenet import Segment, load_segment_wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WENET_S_JSONL))
    ap.add_argument("--out", default=str(S2_WENET_CHAT))
    ap.add_argument("--wav-dir", default=str(S2_WENET_WAVS))
    ap.add_argument("--keep", type=int, default=400)
    ap.add_argument("--pool", type=int, default=2500)
    ap.add_argument("--scan", type=int, default=400000)
    args = ap.parse_args()

    out = Path(args.out)
    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    kept = read_jsonl(out)
    seen_text = {str(r.get("user") or "") for r in kept}
    if len(kept) >= args.keep:
        print(f"already have {len(kept)} >= {args.keep} -> {out}", flush=True)
        return 0

    cands = collect_candidates(Path(args.src), args.pool, args.scan)
    print(f"candidates {len(cands)} kept_so_far {len(kept)}", flush=True)
    model = MiniCPM5Audio.from_pretrained(
        MINICPM5,
        MINICPMO,
        encoder_backend="sensevoice",
        sensevoice_path=SENSEVOICE,
    )
    with out.open("a", encoding="utf-8") as w:
        for row in cands:
            if len(kept) >= args.keep:
                break
            text = str(row["text"]).strip()
            if text in seen_text:
                continue
            try:
                reply = model.generate_from_text(
                    text, tools=None, system=T2T_SYSTEM, max_new_tokens=80
                )
            except Exception as exc:  # noqa: BLE001
                print(f"skip gen {text!r}: {exc}", flush=True)
                continue
            reply = (reply or "").strip()
            if not ok_reply(text, reply):
                print(f"drop {text!r} -> {reply!r}", flush=True)
                continue
            sid = f"wenet_chat_{len(kept):04d}"
            seg = Segment(
                sid=str(row.get("id") or sid),
                wav=Path(str(row["wav"])),
                start=float(row["start"]),
                duration=float(row["duration"]),
                text=text,
            )
            wav = load_segment_wav(seg, sr=16000)
            if wav.size < 1600:
                print(f"drop short wav {text!r}", flush=True)
                continue
            wav_path = wav_dir / f"{sid}.wav"
            peak = float(np.max(np.abs(wav))) or 1.0
            sf.write(str(wav_path), (wav / max(peak, 1e-6) * 0.9).astype(np.float32), 16000)
            rec = {
                "id": sid,
                "wav": rel_wav(wav_path),
                "user": text,
                "target": format_nofc(reply),
                "route": "nofc",
                "kind": "chat",
                "source": "wenet_s",
                "wenet_id": row.get("id"),
                "reply": reply,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()
            kept.append(rec)
            seen_text.add(text)
            print(f"keep {sid} {text!r} -> {reply!r}", flush=True)

    print(f"wrote {len(kept)} -> {out}", flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
