#!/usr/bin/env python3
"""Inference interface for the wave-onset char aligner.

Outputs per-character `{char, onset, start, end}` intervals using the Voronoi
convention: start_i = onset_i, end_i = onset_{i+1}, final end = audio duration.

Usage:
  # single utterance
  python char_align/infer.py --wav /path/x.wav --text "在工作中大"

  # batch over a jsonl (reuses gold text; writes per-record alignments)
  python char_align/infer.py --jsonl data/char_align/s06_part1/test.jsonl \
      --out-json tmp/infer_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import load_jsonl  # noqa: E402
from char_align.model_utils import onset_peaks_onsets  # noqa: E402
from char_align.train_small_wave import (  # noqa: E402
    FRAME_SHIFT_S,
    SAMPLE_RATE,
    SmallOnsetWaveModel,
)

CKPT = "runs/char_align/small_wave/best.pt"
MIN_GAP_S = 0.35


def load_model(device="cuda"):
    model = SmallOnsetWaveModel()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    return model


@torch.no_grad()
def align_chars(model, wav: torch.Tensor, text: str, device="cuda"):
    """wav: (T,) float 16kHz. Returns list of {char, onset, start, end}."""
    chars = list(text)
    n = len(chars)
    logits = model(wav.unsqueeze(0).to(device))[0]  # (Tf,)
    T = logits.size(0)
    onsets = onset_peaks_onsets(logits, n, T, FRAME_SHIFT_S, min_gap_s=MIN_GAP_S)
    duration = wav.numel() / SAMPLE_RATE
    out = []
    for i, ch in enumerate(chars):
        start = onsets[i]
        end = onsets[i + 1] if i + 1 < n else duration
        if end < start:
            end = start
        out.append({"char": ch, "onset": round(start, 3), "start": round(start, 3), "end": round(end, 3)})
    return out


def load_wav(path: str, device="cuda"):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    wav = torch.from_numpy(np.ascontiguousarray(audio))
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_model(args.device)

    if args.wav:
        wav = load_wav(args.wav, args.device)
        items = align_chars(model, wav, args.text, args.device)
        result = {"wav": args.wav, "text": args.text, "chars": items}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {args.out_json}")
        return

    if args.jsonl:
        rows = load_jsonl(Path(args.jsonl))
        out = []
        for r in rows:
            wav = load_wav(r["wav"], args.device)
            items = align_chars(model, wav, r["text"], args.device)
            out.append({"id": r["id"], "text": r["text"], "chars": items})
        text = json.dumps(out, ensure_ascii=False, indent=2)
        if args.out_json:
            Path(args.out_json).write_text(text, encoding="utf-8")
            print(f"wrote {args.out_json} ({len(out)} utterances)")
        else:
            print(text)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
