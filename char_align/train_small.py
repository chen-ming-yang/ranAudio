#!/usr/bin/env python3
"""Train a small from-scratch onset-detection model on S06 gold char alignments.

Unlike the Zipformer fine-tune, this runs directly on 10ms log-mel frames (no
8x downsampling), so onset resolution is bounded by 10ms instead of 80ms.

Usage (from repo root):
  source /cmy/cmy/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr
  python char_align/train_small.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import (  # noqa: E402
    CharAlignDataset,
    collate_align,
    gaussian_onset_targets,
    load_jsonl,
)
from char_align.metrics import AlignStats  # noqa: E402
from char_align.model_utils import onset_monotonic_dp, onset_peaks_onsets  # noqa: E402

FRAME_SHIFT_S = 0.01  # 10ms hop


class SmallOnsetModel(nn.Module):
    """Conv + BiGRU onset detector. Input (B,T,80) log-mel → (B,T) onset logits."""

    def __init__(self, n_mels: int = 80, hidden: int = 128, n_gru_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(hidden)
        self.gru = nn.GRU(
            hidden, hidden, num_layers=n_gru_layers, bidirectional=True, batch_first=True
        )
        self.onset_head = nn.Linear(2 * hidden, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # per-utterance mean/var norm over time
        mu = feats.mean(dim=1, keepdim=True)
        sd = feats.std(dim=1, keepdim=True) + 1e-5
        x = (feats - mu) / sd
        x = x.transpose(1, 2)  # (B,80,T)
        x = self.conv(x).transpose(1, 2)  # (B,T,hidden)
        x = self.ln(x)
        x, _ = self.gru(x)  # (B,T,2h)
        return self.onset_head(x).squeeze(-1)  # (B,T)


def run_epoch(model, loader, optim, device, args, train: bool):
    model.train(train)
    total = {"loss": 0.0, "n": 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            feats = batch["feats"].to(device)
            feat_lens = batch["feat_lens"].to(device)
            onsets = batch["onsets"].to(device)
            char_lens = batch["char_lens"].to(device)
            logits = model(feats)  # (B,T)
            T = logits.size(1)
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
            total["loss"] += float(loss.item()) * feats.size(0)
            total["n"] += feats.size(0)
    return total["loss"] / max(total["n"], 1)


@torch.no_grad()
def predict_onsets(model, feats, feat_len, n_chars, device, decode):
    logits = model(feats.unsqueeze(0).to(device))[0]  # (T,)
    T = int(feat_len)
    if decode == "dp":
        return onset_monotonic_dp(logits, n_chars, T, FRAME_SHIFT_S)
    return onset_peaks_onsets(logits, n_chars, T, FRAME_SHIFT_S)


@torch.no_grad()
def evaluate(model, ds, device, decode):
    model.eval()
    stats = AlignStats()
    for i in range(len(ds)):
        item = ds[i]
        feat = item["feat"]  # (T,80) log-mel
        n_chars = int(item["onsets"].numel())
        pred = predict_onsets(model, feat, feat.size(0), n_chars, device, decode)
        gold = item["onsets"].tolist()
        stats.add_utt(gold, pred)
    out = stats.summary()
    out["decode"] = decode
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/char_align/s06_part1/train.jsonl")
    ap.add_argument("--dev-jsonl", default="data/char_align/s06_part1/dev.jsonl")
    ap.add_argument("--test-jsonl", default="data/char_align/s06_part1/test.jsonl")
    ap.add_argument("--out-dir", default="runs/char_align/small_onset")
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    train_ds = CharAlignDataset(args.train_jsonl, "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model")
    dev_ds = CharAlignDataset(args.dev_jsonl, "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model")
    test_ds = CharAlignDataset(args.test_jsonl, "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model")
    if args.overfit_n > 0:
        train_ds.rows = train_ds.rows[: args.overfit_n]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_align,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_align,
    )

    model = SmallOnsetModel(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.1f}k")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_mae = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optim, device, args, True)
        dv_loss = run_epoch(model, dev_loader, optim, device, args, False)
        dv = evaluate(model, dev_ds, device, decode="topn")
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

    # final eval on best
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    results = {"n_train": len(train_ds.rows), "n_dev": len(dev_ds.rows), "n_test": len(test_ds.rows)}
    results["dev_topn"] = evaluate(model, dev_ds, device, "topn")
    results["dev_dp"] = evaluate(model, dev_ds, device, "dp")
    results["test_topn"] = evaluate(model, test_ds, device, "topn")
    results["test_dp"] = evaluate(model, test_ds, device, "dp")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
