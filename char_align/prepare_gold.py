#!/usr/bin/env python3
"""Extract S06 gold char-onset annotations from tar → stratified train/dev/test jsonl.

Usage:
  python char_align/prepare_gold.py \
    --tar /cmy/cmy/rmsaqa-code/同事新标注_06_part1_合并.tar \
    --out-dir data/char_align/s06_part1
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
import wave
from collections import defaultdict
from io import BytesIO
from pathlib import Path


def wav_duration_sec(raw: bytes) -> float:
    with wave.open(BytesIO(raw), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def task_id(utt: str) -> str:
    # S006T004E000N00000 → T004
    if len(utt) >= 8 and utt[4] == "T":
        return utt[4:8]
    return "TUNK"


def convert_one(obj: dict, duration: float) -> dict | None:
    """onset time → [start,end]; drop utt if any null time."""
    anns = obj.get("annotations") or []
    if not anns:
        return None
    times = []
    chars = []
    for a in anns:
        t = a.get("time")
        ch = a.get("char")
        if t is None or ch is None:
            return None
        times.append(float(t))
        chars.append(str(ch))
    text = obj.get("text") or "".join(chars)
    if "".join(chars) != text:
        return None
    # monotonic soft fix: allow tiny jitter
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            times[i] = times[i - 1]
    items = []
    for i, ch in enumerate(chars):
        start = times[i]
        end = times[i + 1] if i + 1 < len(times) else float(duration)
        if end < start:
            end = start
        items.append(
            {
                "char": ch,
                "onset": start,
                "start": start,
                "end": end,
            }
        )
    utt = Path(obj["audio"]).stem
    return {
        "id": utt,
        "audio": obj["audio"],
        "text": text,
        "duration": float(duration),
        "task": task_id(utt),
        "chars": items,
    }


def stratified_split(rows: list[dict], seed: int, ratios=(0.8, 0.1, 0.1)):
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    rng = random.Random(seed)
    train, dev, test = [], [], []
    for task, items in sorted(by_task.items()):
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            train.extend(items)
            continue
        if n == 2:
            train.append(items[0])
            test.append(items[1])
            continue
        n_test = max(1, int(round(n * ratios[2])))
        n_dev = max(1, int(round(n * ratios[1])))
        while n_test + n_dev >= n:
            if n_dev > 1:
                n_dev -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        test.extend(items[:n_test])
        dev.extend(items[n_test : n_test + n_dev])
        train.extend(items[n_test + n_dev :])
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tar",
        default="/cmy/cmy/rmsaqa-code/同事新标注_06_part1_合并.tar",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/char_align/s06_part1"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extract-audio", action="store_true", default=True)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    audio_dir = out_dir / "Audio" / "06"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dropped = 0
    with tarfile.open(args.tar, "r") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        ann_names = sorted(n for n in members if "/Annotation/" in n and n.endswith(".json"))
        for an in ann_names:
            obj = json.load(tf.extractfile(members[an]))
            wav_name = obj["audio"]
            # prefer packed audio
            cand = [
                f"同事新标注_06_part1_合并/Audio/06/{wav_name}",
                f"同事新标注_06_part1_合并/Audio/{wav_name}",
            ]
            raw = None
            src = None
            for c in cand:
                if c in members:
                    raw = tf.extractfile(members[c]).read()
                    src = c
                    break
            if raw is None:
                dropped += 1
                continue
            duration = wav_duration_sec(raw)
            rec = convert_one(obj, duration)
            if rec is None:
                dropped += 1
                continue
            dest = audio_dir / wav_name
            if args.extract_audio and not dest.exists():
                dest.write_bytes(raw)
            rec["wav"] = str(dest.resolve())
            rows.append(rec)

    train, dev, test = stratified_split(rows, seed=args.seed)
    for name, split in ("train", train), ("dev", dev), ("test", test):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} → {path}")

    meta = {
        "n_kept": len(rows),
        "n_dropped": dropped,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "seed": args.seed,
        "tar": args.tar,
        "tasks": sorted({r["task"] for r in rows}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
