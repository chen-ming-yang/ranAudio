#!/usr/bin/env python3
"""Evaluate the small from-scratch onset model on dev/test, tuning decode params."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import CharAlignDataset  # noqa: E402
from char_align.metrics import AlignStats  # noqa: E402
from char_align.model_utils import onset_peaks_onsets  # noqa: E402
from char_align.train_small import FRAME_SHIFT_S, SmallOnsetModel  # noqa: E402

TOK = "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model"


@torch.no_grad()
def evaluate(model, ds, device, min_gap_s):
    model.eval()
    stats = AlignStats()
    for i in range(len(ds)):
        item = ds[i]
        feat = item["feat"].to(device).unsqueeze(0)
        logits = model(feat)[0]
        n = int(item["onsets"].numel())
        pred = onset_peaks_onsets(logits, n, feat.size(1), FRAME_SHIFT_S, min_gap_s=min_gap_s)
        gold = item["onsets"].tolist()
        stats.add_utt(gold, pred)
    out = stats.summary()
    out["min_gap_s"] = min_gap_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/char_align/s06_part1")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--checkpoint", default="runs/char_align/small_onset/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-gap-s", type=float, default=0.05)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    ds = CharAlignDataset(Path(args.data_dir) / f"{args.split}.jsonl", TOK)
    model = SmallOnsetModel()
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
