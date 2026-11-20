#!/usr/bin/env python3
"""Waveform-input small onset model — ablation vs mel-input version.

Same BiGRU + onset head as train_small.py, but the input is raw 16kHz waveform
instead of 80-dim log-mel. A learnable Conv1d front-end downsamples the waveform
to a 10ms frame rate (kernel=400=25ms, stride=160=10ms), exactly matching the
MelSpectrogram hop, so frame counts / gold frame_shift are identical.

Usage (from repo root):
  source /cmy/cmy/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr
  python char_align/train_small_wave.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import (  # noqa: E402
    gaussian_onset_targets,
    load_jsonl,
)
from char_align.metrics import AlignStats  # noqa: E402
from char_align.model_utils import onset_peaks_onsets  # noqa: E402

FRAME_SHIFT_S = 0.01  # 10ms: stride 160 samples @ 16kHz
SAMPLE_RATE = 16000
STRIDE = 160   # samples per frame
KERNEL = 400   # 25ms window


class WaveDataset(Dataset):
    """Loads raw 16kHz waveform + gold onsets (no mel computation)."""

    def __init__(self, jsonl: Path):
        self.rows = load_jsonl(Path(jsonl))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        audio, sr = sf.read(r["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(audio))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        onsets = torch.tensor([c["onset"] for c in r["chars"]], dtype=torch.float)
        return {
            "id": r["id"],
            "wav": wav,
            "onsets": onsets,
            "n_char": onsets.numel(),
            "n_sample": wav.numel(),
            "text": r["text"],
        }


def collate_wave(batch: list[dict]) -> dict:
    B = len(batch)
    L = max(b["n_sample"] for b in batch)
    Cmax = max(b["n_char"] for b in batch)
    wavs = torch.zeros(B, L)
    onsets = torch.full((B, Cmax), -1.0)
    char_lens = torch.zeros(B, dtype=torch.long)
    n_samples = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        l = b["n_sample"]
        wavs[i, :l] = b["wav"]
        n_samples[i] = l
        n = b["n_char"]
        onsets[i, :n] = b["onsets"]
        char_lens[i] = n
    return {"wavs": wavs, "n_samples": n_samples, "onsets": onsets, "char_lens": char_lens}


def wave_feat_lens(n_samples: torch.Tensor) -> torch.Tensor:
    """Front-end output frames per utterance: (L + 2*pad - K)//S + 1, pad=K//2."""
    return (n_samples + 2 * (KERNEL // 2) - KERNEL) // STRIDE + 1


class SmallOnsetWaveModel(nn.Module):
    """Learnable Conv front-end (waveform → 10ms) + BiGRU + onset head."""

    def __init__(self, hidden: int = 128, n_gru_layers: int = 2):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(1, hidden, KERNEL, stride=STRIDE, padding=KERNEL // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=n_gru_layers, bidirectional=True, batch_first=True
        )
        self.onset_head = nn.Linear(2 * hidden, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # per-utterance normalization over samples
        mu = wav.mean(dim=1, keepdim=True)
        sd = wav.std(dim=1, keepdim=True) + 1e-5
        x = (wav - mu) / sd
        x = x.unsqueeze(1)  # (B,1,L)
        x = self.front(x).transpose(1, 2)  # (B,T,hidden)
        x = self.ln(x)
        x, _ = self.gru(x)  # (B,T,2h)
        return self.onset_head(x).squeeze(-1)  # (B,T)


def run_epoch(model, loader, optim, device, args, train: bool):
    model.train(train)
    total = {"loss": 0.0, "n": 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            wavs = batch["wavs"].to(device)
            n_samples = batch["n_samples"].to(device)
            onsets = batch["onsets"].to(device)
            char_lens = batch["char_lens"].to(device)
            logits = model(wavs)  # (B,T)
            T = logits.size(1)
            feat_lens = wave_feat_lens(n_samples).clamp(min=1, max=T)
            tgt = gaussian_onset_targets(
                onsets, char_lens, feat_lens, FRAME_SHIFT_S, sigma_s=args.sigma_s, time_steps=T
            )
            mask = torch.arange(T, device=device).unsqueeze(0) < feat_lens.unsqueeze(1)
            loss = F.binary_cross_entropy_with_logits(logits[mask], tgt[mask], reduction="mean")
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total["loss"] += float(loss.item()) * wavs.size(0)
            total["n"] += wavs.size(0)
    return total["loss"] / max(total["n"], 1)


@torch.no_grad()
def predict_onsets(model, wav, n_chars, device, min_gap_s):
    logits = model(wav.unsqueeze(0).to(device))[0]
    T = logits.size(0)
    return onset_peaks_onsets(logits, n_chars, T, FRAME_SHIFT_S, min_gap_s=min_gap_s)


@torch.no_grad()
def evaluate(model, ds, device, min_gap_s):
    model.eval()
    stats = AlignStats()
    for i in range(len(ds)):
        item = ds[i]
        pred = predict_onsets(model, item["wav"], item["n_char"], device, min_gap_s)
        stats.add_utt(item["onsets"].tolist(), pred)
    out = stats.summary()
    out["min_gap_s"] = min_gap_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/char_align/s06_part1/train.jsonl")
    ap.add_argument("--dev-jsonl", default="data/char_align/s06_part1/dev.jsonl")
    ap.add_argument("--test-jsonl", default="data/char_align/s06_part1/test.jsonl")
    ap.add_argument("--out-dir", default="runs/char_align/small_wave")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma-s", type=float, default=0.03)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overfit-n", type=int, default=0)
    ap.add_argument("--min-gap-s", type=float, default=0.35)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    train_ds = WaveDataset(args.train_jsonl)
    dev_ds = WaveDataset(args.dev_jsonl)
    test_ds = WaveDataset(args.test_jsonl)
    if args.overfit_n > 0:
        train_ds.rows = train_ds.rows[: args.overfit_n]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_wave,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_wave,
    )

    model = SmallOnsetWaveModel(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.1f}k")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_mae = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optim, device, args, True)
        dv_loss = run_epoch(model, dev_loader, optim, device, args, False)
        dv = evaluate(model, dev_ds, device, args.min_gap_s)
        mae = dv["onset_mae_s"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "dev_loss": dv_loss, "dev_onset_mae_s": mae})
        print(
            f"epoch {epoch}: train_loss={tr_loss:.4f} dev_loss={dv_loss:.4f} "
            f"dev_onset_mae={mae:.4f}s"
        )
        ckpt = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(ckpt, out_dir / "last.pt")
        if mae < best_mae:
            best_mae = mae
            no_improve = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best.pt (dev_onset_mae={best_mae:.4f}s)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop at epoch {epoch} (patience={args.patience})")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    results = {"n_train": len(train_ds.rows), "n_dev": len(dev_ds.rows), "n_test": len(test_ds.rows)}
    results["dev"] = evaluate(model, dev_ds, device, args.min_gap_s)
    results["test"] = evaluate(model, test_ds, device, args.min_gap_s)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
