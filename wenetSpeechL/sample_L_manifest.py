#!/usr/bin/env python3
"""Sample a fraction of WenetSpeech L Lhotse supervisions into a flat JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def stable_bucket(utt_id: str, buckets: int) -> int:
    """Deterministic bucket in [0, buckets)."""
    h = hashlib.md5(utt_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % buckets


def load_recordings(rec_path: Path) -> dict[str, str]:
    id_to_path: dict[str, str] = {}
    with rec_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["id"]
            sources = rec.get("sources") or []
            if not sources:
                continue
            id_to_path[rid] = sources[0]["source"]
    return id_to_path


def detect_source(audio_path: str) -> str:
    p = audio_path.lower()
    if "/youtube/" in p or "youtube" in p:
        return "youtube"
    if "/podcast/" in p or "podcast" in p:
        return "podcast"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sup",
        type=Path,
        default=Path("/cmy/cmy/real-asr/data/manifests/wenetspeech/train/supervisions.jsonl"),
    )
    parser.add_argument(
        "--rec",
        type=Path,
        default=Path("/cmy/cmy/real-asr/data/manifests/wenetspeech/train/recordings.jsonl"),
    )
    parser.add_argument("--fraction", type=float, default=0.1, help="Keep fraction of utts (default 0.1)")
    parser.add_argument(
        "--complement",
        action="store_true",
        help="Keep the complement of --fraction (e.g. fraction=0.1 -> remaining ~90%%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Used only for logging; sampling is hash-based")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/sample_manifest.jsonl"),
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Cap output lines (0=all selected)")
    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit("--fraction must be in (0, 1]")

    # Keep first K of 1000 buckets ≈ fraction * 1000; --complement keeps the rest
    buckets = 1000
    keep_n = max(1, int(round(args.fraction * buckets)))
    print(f"Loading recordings from {args.rec} ...", flush=True)
    id_to_path = load_recordings(args.rec)
    print(f"  recordings={len(id_to_path)}", flush=True)
    print(
        f"  mode={'complement' if args.complement else 'fraction'} "
        f"fraction={args.fraction} keep_buckets={keep_n}/{buckets}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    missing_rec = 0

    with args.sup.open("r", encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            if total % 1_000_000 == 0:
                print(f"  scanned={total} kept={kept}", flush=True)

            sup = json.loads(line)
            utt_id = sup["id"]
            b = stable_bucket(utt_id, buckets)
            # fraction: bucket < keep_n; complement: bucket >= keep_n
            if args.complement:
                if b < keep_n:
                    continue
            else:
                if b >= keep_n:
                    continue

            rid = sup["recording_id"]
            audio_path = id_to_path.get(rid)
            if not audio_path:
                missing_rec += 1
                continue

            custom = sup.get("custom") or {}
            row = {
                "utt_id": utt_id,
                "recording_id": rid,
                "audio_path": audio_path,
                "start": float(sup["start"]),
                "duration": float(sup["duration"]),
                "text": (sup.get("text") or "").strip(),
                "confidence": float(custom.get("confidence", -1.0)),
                "subsets": custom.get("subsets") or [],
                "source": detect_source(audio_path),
                "seed": args.seed,
                "fraction": args.fraction,
                "complement": bool(args.complement),
            }
            if not row["text"]:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_samples > 0 and kept >= args.max_samples:
                break

    print(
        json.dumps(
            {
                "total_scanned": total,
                "kept": kept,
                "missing_recording": missing_rec,
                "fraction": args.fraction,
                "complement": bool(args.complement),
                "keep_buckets": keep_n,
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
