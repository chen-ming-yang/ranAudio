#!/usr/bin/env python3
"""Fine-tune Zipformer CTC + onset head on S06 gold char alignments.

Usage (from repo root):
  source /cmy/cmy/miniconda3/etc/profile.d/conda.sh && conda activate qwen3-asr
  PYTHONPATH=/cmy/cmy/real-asr:$PWD python char_align/train_align.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from char_align.dataset import (  # noqa: E402
    BLANK_ID,
    CharAlignDataset,
    build_frame_targets,
    collate_align,
    gaussian_onset_targets,
)
from char_align.model_utils import (  # noqa: E402
    encoder_frame_shift_s,
    load_asr_bundle,
    onset_monotonic_dp,
    onset_peaks_onsets,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/char_align/s06_part1/train.jsonl")
    ap.add_argument("--dev-jsonl", default="data/char_align/s06_part1/dev.jsonl")
    ap.add_argument("--tokenizer", default="/cmy/cmy/real-asr/data/tokenizer/sp_6000.model")
    ap.add_argument("--config", default="/cmy/cmy/real-asr/configs/train_finetune_dysarthric.yaml")
    ap.add_argument("--checkpoint", default="/cmy/cmy/real-asr/exp/finetune_dysarthric/epoch_30.pt")
    ap.add_argument("--out-dir", default="runs/char_align/s06_part1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=3e-6)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--ctc-weight", type=float, default=0.2)
    ap.add_argument("--onset-weight", type=float, default=0.5)
    ap.add_argument("--sigma-s", type=float, default=0.04)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5, help="early stop after N epochs w/o dev onset MAE improvement")
    ap.add_argument("--overfit-n", type=int, default=0, help="if >0, only first N train utts")
    return ap.parse_args()


def run_epoch(
    transducer,
    ctc_head,
    onset_head,
    loader,
    optim,
    device,
    frame_shift_s,
    args,
    train: bool,
):
    transducer.train(train)
    ctc_head.train(train)
    onset_head.train(train)
    total = {"loss": 0.0, "ce": 0.0, "ctc": 0.0, "onset": 0.0, "n": 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            feats = batch["feats"].to(device)
            feat_lens = batch["feat_lens"].to(device)
            tokens = batch["tokens"].to(device)
            token_lens = batch["token_lens"].to(device)
            onsets = batch["onsets"].to(device)
            starts = batch["starts"].to(device)
            ends = batch["ends"].to(device)
            char_lens = batch["char_lens"].to(device)

            enc, enc_lens = transducer.encoder(feats, feat_lens)
            # icefall out_lens can disagree with tensor length by 1
            T_enc = enc.size(1)
            enc_lens = enc_lens.clamp(min=1, max=T_enc)
            logits = ctc_head.get_logits(enc)  # (B,T,V)
            log_probs = F.log_softmax(logits, dim=-1)
            B, T, V = logits.shape
            assert T == T_enc

            frame_tgt = build_frame_targets(
                starts, ends, char_lens, tokens, enc_lens, frame_shift_s, blank_id=BLANK_ID
            )
            if frame_tgt.size(1) < T:
                pad = torch.full(
                    (B, T - frame_tgt.size(1)),
                    BLANK_ID,
                    dtype=frame_tgt.dtype,
                    device=device,
                )
                frame_tgt = torch.cat([frame_tgt, pad], dim=1)
            elif frame_tgt.size(1) > T:
                frame_tgt = frame_tgt[:, :T]
            # mask padded encoder frames
            mask = torch.arange(T, device=device).unsqueeze(0) < enc_lens.unsqueeze(1)
            ce = F.cross_entropy(
                logits[mask],
                frame_tgt[mask],
                reduction="mean",
            )

            # CTC expects (T,B,V)
            ctc = F.ctc_loss(
                log_probs.transpose(0, 1),
                tokens,
                enc_lens,
                token_lens,
                blank=BLANK_ID,
                zero_infinity=True,
            )

            onset_logits = onset_head(enc)
            onset_tgt = gaussian_onset_targets(
                onsets,
                char_lens,
                enc_lens,
                frame_shift_s,
                sigma_s=args.sigma_s,
                time_steps=T,
            )
            bce = F.binary_cross_entropy_with_logits(
                onset_logits[mask], onset_tgt[mask], reduction="mean"
            )

            loss = args.ce_weight * ce + args.ctc_weight * ctc + args.onset_weight * bce

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(transducer.parameters())
                    + list(ctc_head.parameters())
                    + list(onset_head.parameters()),
                    1.0,
                )
                optim.step()

            bs = feats.size(0)
            total["loss"] += float(loss.item()) * bs
            total["ce"] += float(ce.item()) * bs
            total["ctc"] += float(ctc.item()) * bs
            total["onset"] += float(bce.item()) * bs
            total["n"] += bs
    n = max(total["n"], 1)
    return {k: (total[k] / n if k != "n" else total[k]) for k in total}


@torch.no_grad()
def dev_onset_mae(transducer, onset_head, loader, device, frame_shift_s):
    """Onset-head peak-picking MAE (seconds) on dev set — the real metric we tune."""
    transducer.eval()
    onset_head.eval()
    errs = []
    for batch in loader:
        feats = batch["feats"].to(device)
        feat_lens = batch["feat_lens"].to(device)
        onsets = batch["onsets"].to(device)  # (B, Cmax) with -1 padding
        char_lens = batch["char_lens"].to(device)
        enc, enc_lens = transducer.encoder(feats, feat_lens)
        olog = onset_head(enc)  # (B, T)
        for b in range(feats.size(0)):
            n = int(char_lens[b].item())
            if n == 0:
                continue
            elen = int(enc_lens[b].item())
            pred = onset_monotonic_dp(olog[b], n, elen, frame_shift_s)
            gold = [float(onsets[b, i].item()) for i in range(n) if onsets[b, i] >= 0]
            if len(gold) != len(pred):
                continue
            for g, p in zip(gold, pred):
                errs.append(abs(g - p))
    return (sum(errs) / len(errs)) if errs else float("inf")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, transducer, ctc_head, onset_head = load_asr_bundle(
        config=args.config, checkpoint=args.checkpoint, device=args.device
    )
    frame_shift_s = encoder_frame_shift_s(transducer, device=args.device)
    print(f"estimated frame_shift_s={frame_shift_s:.4f}")

    train_ds = CharAlignDataset(args.train_jsonl, args.tokenizer)
    if args.overfit_n > 0:
        train_ds.rows = train_ds.rows[: args.overfit_n]
    dev_ds = CharAlignDataset(args.dev_jsonl, args.tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_align,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_align,
    )

    # differential LRs
    enc_params = list(transducer.encoder.parameters())
    head_params = list(ctc_head.parameters()) + list(onset_head.parameters())
    # also allow joiner/predictor freeze — only encoder in transducer for CE path
    for n, p in transducer.named_parameters():
        if not n.startswith("encoder"):
            p.requires_grad = False

    optim = torch.optim.AdamW(
        [
            {"params": [p for p in enc_params if p.requires_grad], "lr": args.lr_enc},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )

    best_mae = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            transducer, ctc_head, onset_head, train_loader, optim, args.device, frame_shift_s, args, True
        )
        dv = run_epoch(
            transducer, ctc_head, onset_head, dev_loader, optim, args.device, frame_shift_s, args, False
        )
        mae = dev_onset_mae(transducer, onset_head, dev_loader, args.device, frame_shift_s)
        row = {
            "epoch": epoch,
            "train": tr,
            "dev": dv,
            "dev_onset_mae_s": mae,
            "frame_shift_s": frame_shift_s,
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={tr['loss']:.4f} ce={tr['ce']:.4f} "
            f"ctc={tr['ctc']:.4f} onset={tr['onset']:.4f} | "
            f"dev_loss={dv['loss']:.4f} dev_onset_mae={mae:.4f}s"
        )
        ckpt = {
            "model": transducer.state_dict(),
            "ctc_head": ctc_head.state_dict(),
            "onset_head": onset_head.state_dict(),
            "epoch": epoch,
            "frame_shift_s": frame_shift_s,
            "args": vars(args),
        }
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
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
