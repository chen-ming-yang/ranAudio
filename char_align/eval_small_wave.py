#!/usr/bin/env python3
"""Evaluate the waveform-input small onset model, sweeping min_gap_s."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.train_small_wave import (  # noqa: E402
    SmallOnsetWaveModel,
    WaveDataset,
    evaluate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/char_align/s06_part1")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--checkpoint", default="runs/char_align/small_wave/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-gap-s", type=float, default=0.35)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    ds = WaveDataset(Path(args.data_dir) / f"{args.split}.jsonl")
    model = SmallOnsetWaveModel()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device)

    out = evaluate(model, ds, args.device, args.min_gap_s)
    out["split"] = args.split
    out["checkpoint"] = args.checkpoint
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
