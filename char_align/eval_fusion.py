#!/usr/bin/env python3
"""Fuse CTC forced-align (text sequence constraint) with onset-head peak (10ms
precision).

Idea: CTC forced-align gives one coarse onset per char with strict ordering
(monotonic), but at 80ms frame shift. The wave onset head gives a per-frame onset
probability at 10ms. We use the CTC onsets as *ordering anchors*: cut the time
axis into N intervals at the midpoints between adjacent CTC onsets, then within
each interval pick the onset-probability argmax as the refined onset.

This keeps the char count / ordering from CTC while getting 10ms timing from the
onset head.

Usage:
  PYTHONPATH=/cmy/cmy/real-asr:$PWD python char_align/eval_fusion.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import char_token_ids, load_jsonl  # noqa: E402
from char_align.metrics import AlignStats  # noqa: E402
from char_align.model_utils import (  # noqa: E402
    encoder_frame_shift_s,
    forced_align_onsets,
    forced_align_spans,
    load_asr_bundle,
    onset_peaks_onsets,
)
from char_align.train_small_wave import (  # noqa: E402
    FRAME_SHIFT_S,
    SAMPLE_RATE,
    SmallOnsetWaveModel,
)

TOK = "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model"
CONFIG = "/cmy/cmy/real-asr/configs/train_finetune_dysarthric.yaml"
BASE_CKPT = "/cmy/cmy/real-asr/exp/finetune_dysarthric/epoch_30.pt"
WAVE_CKPT = "runs/char_align/small_wave/best.pt"


def load_all(device):
    cfg, transducer, ctc_head, _ = load_asr_bundle(
        config=CONFIG, checkpoint=BASE_CKPT, device=device
    )
    transducer.eval(); ctc_head.eval()
    fss = encoder_frame_shift_s(transducer, device=device)

    wave = SmallOnsetWaveModel()
    ck = torch.load(WAVE_CKPT, map_location="cpu", weights_only=False)
    wave.load_state_dict(ck["model"])
    wave = wave.to(device).eval()

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160, n_mels=80
    ).to(device)

    sp = spm.SentencePieceProcessor()
    sp.Load(TOK)
    return transducer, ctc_head, wave, fbank, sp, fss


@torch.no_grad()
def run(rows, device):
    transducer, ctc_head, wave, fbank, sp, fss = load_all(device)

    stats_ctc = AlignStats()
    stats_wave = AlignStats()
    stats_fuse = AlignStats()
    for r in rows:
        audio, sr = sf.read(r["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(audio))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.to(device)
        text = r["text"]
        gold = [c["onset"] for c in r["chars"]]
        n = len(gold)

        # --- CTC forced-align coarse onsets (80ms) ---
        feat = fbank(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
        feat = torch.log(feat.clamp(min=1e-5)).unsqueeze(0)
        feat_lens = torch.tensor([feat.size(1)], device=device)
        enc, enc_lens = transducer.encoder(feat, feat_lens)
        log_probs = ctc_head(enc)[0]
        tids = torch.tensor(char_token_ids(sp, text), device=device)
        elen = int(enc_lens[0].item())
        t_ctc = forced_align_onsets(
            log_probs, tids, elen, fss, blank_id=0
        )  # seconds, length n
        stats_ctc.add_utt(gold, t_ctc)

        # --- wave onset probability (10ms) ---
        wave_logits = wave(wav.unsqueeze(0))[0]
        p = torch.sigmoid(wave_logits).cpu().numpy().astype(float)
        T = p.size
        # pure onset-head decode (baseline, greedy NMS)
        wave_pred = onset_peaks_onsets(
            wave_logits, n, T, FRAME_SHIFT_S, min_gap_s=0.35
        )
        stats_wave.add_utt(gold, wave_pred)

        # --- fusion: cut intervals at CTC onset midpoints, argmax within ---
        # CTC onset = the onset anchor; interval i spans [mid(prev,cur), mid(cur,next)]
        onsets_fuse = []
        for i in range(n):
            lo_i = (t_ctc[i - 1] + t_ctc[i]) / 2 if i > 0 else 0.0
            hi_i = (t_ctc[i] + t_ctc[i + 1]) / 2 if i + 1 < n else T * FRAME_SHIFT_S
            lo = max(0, min(T - 1, int(round(lo_i / FRAME_SHIFT_S))))
            hi = max(0, min(T - 1, int(round(hi_i / FRAME_SHIFT_S))))
            if hi < lo:
                hi = lo
            seg = p[lo:hi + 1]
            best = int(np.argmax(seg)) + lo
            onsets_fuse.append((best + 0.5) * FRAME_SHIFT_S)

        stats_fuse.add_utt(gold, onsets_fuse)

    return {
        "ctc_fa": stats_ctc.summary(),
        "wave_onset": stats_wave.summary(),
        "fusion": stats_fuse.summary(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/char_align/s06_part1")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl")
    out = run(rows, args.device)
    out["split"] = args.split
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
