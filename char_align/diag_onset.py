#!/usr/bin/env python3
"""Diagnose onset-head probability curve vs gold onsets on a few dev samples."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import load_jsonl  # noqa: E402
from char_align.model_utils import load_asr_bundle  # noqa: E402

DEV = "data/char_align/s06_part1/dev.jsonl"
CKPT = "runs/char_align/s06_part1_v2/best.pt"
CONFIG = "/cmy/cmy/real-asr/configs/train_finetune_dysarthric.yaml"
BASE = "/cmy/cmy/real-asr/exp/finetune_dysarthric/epoch_30.pt"


def main():
    rows = load_jsonl(Path(DEV))
    cfg, transducer, ctc_head, onset_head = load_asr_bundle(config=CONFIG, checkpoint=BASE, device="cuda")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    transducer.load_state_dict(ckpt["model"], strict=False)
    ctc_head.load_state_dict(ckpt["ctc_head"], strict=False)
    onset_head.load_state_dict(ckpt["onset_head"], strict=False)
    fss = float(ckpt.get("frame_shift_s") or 0.08)
    transducer.eval(); ctc_head.eval(); onset_head.eval()

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80
    ).to("cuda")

    for r in rows[:4]:
        audio, sr = sf.read(r["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(audio)).to("cuda")
        feat = fbank(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
        feat = torch.log(feat.clamp(min=1e-5)).unsqueeze(0)
        lens = torch.tensor([feat.size(1)], device="cuda")
        enc, enc_lens = transducer.encoder(feat, lens)
        olog = onset_head(enc)[0]  # (T,)
        p = torch.sigmoid(olog).detach().cpu().numpy()
        T = int(enc_lens[0].item())
        p = p[:T]
        gold = [c["onset"] for c in r["chars"]]
        gold_frames = [int(round(g / fss)) for g in gold]
        print(f"\n=== {r['id']} text={r['text']!r} dur~{T*fss:.2f}s n_char={len(gold)}")
        print(f"  gold onsets (s): {[round(g,3) for g in gold]}")
        # top-5 prob frames
        top = np.argsort(-p)[:8]
        print(f"  top prob frames: {[(int(t), round(float(p[t]),3), round(t*fss,3)) for t in sorted(top)]}")
        # prob at gold frames
        print(f"  prob at gold frames: {[round(float(p[min(t, T-1)]),3) for t in gold_frames]}")
        print(f"  prob mean={p.mean():.3f} max={p.max():.3f}")


if __name__ == "__main__":
    main()
