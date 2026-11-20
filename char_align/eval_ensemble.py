#!/usr/bin/env python3
"""Ensemble the mel-input and waveform-input small onset models.

Both models emit per-frame onset probability at a 10ms frame rate (mel hop=160,
wave stride=160), so their probability curves align frame-for-frame. We average
the two probability curves (weight alpha for mel, 1-alpha for wave) then run the
same greedy-NMS peak picking.

Usage:
  python char_align/eval_ensemble.py --split test --alpha 0.5
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

from char_align.dataset import CharAlignDataset  # noqa: E402
from char_align.metrics import AlignStats  # noqa: E402
from char_align.model_utils import onset_peaks_onsets  # noqa: E402
from char_align.train_small import FRAME_SHIFT_S, SmallOnsetModel  # noqa: E402
from char_align.train_small_wave import SmallOnsetWaveModel  # noqa: E402

TOK = "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model"
SAMPLE_RATE = 16000


def load_models(device):
    mel = SmallOnsetModel()
    ck = torch.load("runs/char_align/small_onset/best.pt", map_location="cpu", weights_only=False)
    mel.load_state_dict(ck["model"])
    mel = mel.to(device).eval()

    wave = SmallOnsetWaveModel()
    ck = torch.load("runs/char_align/small_wave/best.pt", map_location="cpu", weights_only=False)
    wave.load_state_dict(ck["model"])
    wave = wave.to(device).eval()

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_mels=80
    ).to(device)
    return mel, wave, fbank


@torch.no_grad()
def mel_prob(mel_model, fbank, wav, device):
    feat = fbank(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
    feat = torch.log(feat.clamp(min=1e-5)).unsqueeze(0)
    return torch.sigmoid(mel_model(feat)[0])


@torch.no_grad()
def wave_prob(wave_model, wav, device):
    return torch.sigmoid(wave_model(wav.unsqueeze(0))[0])


@torch.no_grad()
def evaluate(ds, device, alpha, min_gap_s):
    mel_m, wave_m, fbank = load_models(device)
    stats = AlignStats()
    for i in range(len(ds)):
        item = ds[i]
        wav = item["wav"].to(device)
        pm = mel_prob(mel_m, fbank, wav, device)
        pw = wave_prob(wave_m, wav, device)
        T = min(pm.size(0), pw.size(0))
        p = alpha * pm[:T] + (1 - alpha) * pw[:T]
        n = int(item["onsets"].numel())
        pred = onset_peaks_onsets(p, n, T, FRAME_SHIFT_S, min_gap_s=min_gap_s)
        stats.add_utt(item["onsets"].tolist(), pred)
    out = stats.summary()
    out["alpha"] = alpha
    out["min_gap_s"] = min_gap_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/char_align/s06_part1")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-gap-s", type=float, default=0.35)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    ds = WaveDS(args.data_dir, args.split)
    out = evaluate(ds, args.device, args.alpha, args.min_gap_s)
    out["split"] = args.split
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out_json}")


class WaveDS:
    """Lightweight loader: raw 16k wav + gold onsets, indexed like the jsonl."""

    def __init__(self, data_dir, split):
        from char_align.dataset import load_jsonl
        self.rows = load_jsonl(Path(data_dir) / f"{split}.jsonl")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        audio, sr = sf.read(r["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(audio))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        onsets = torch.tensor([c["onset"] for c in r["chars"]], dtype=torch.float)
        return {"wav": wav, "onsets": onsets}


if __name__ == "__main__":
    main()
