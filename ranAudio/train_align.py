#!/usr/bin/env python3
"""Freeze MiniCPM5 + APM; train projector on car ASR transcripts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import torch
from torch.optim import AdamW
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio, freeze_except_projector, save_projector
from ranAudio.paths import DATA_DIR, MINICPM5, MINICPMO, RUNS_DIR, resolve_wav


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(DATA_DIR / "car_audio.jsonl"))
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--out", default=str(RUNS_DIR / "align" / "projector.pt"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = load_jsonl(Path(args.samples))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"align samples={len(samples)}", flush=True)

    model = MiniCPM5Audio.from_pretrained(args.llm, args.apm, encoder_backend="whisper")
    model.train()
    trainable = freeze_except_projector(model)
    opt = AdamW(trainable, lr=args.lr)
    rng = random.Random(args.seed)
    step = 0
    for epoch in range(args.epochs):
        order = list(samples)
        rng.shuffle(order)
        pbar = tqdm(order, desc=f"align {epoch+1}/{args.epochs}")
        for row in pbar:
            wav, _ = librosa.load(str(resolve_wav(row["wav"])), sr=16000, mono=True)
            batch = model.teacher_force_batch(wav, assistant=str(row["text"]), tools=None, system="请把用户语音转成中文文本，只输出转写。")
            if batch["n_label_tokens"] < 1:
                continue
            opt.zero_grad(set_to_none=True)
            out = model.llm(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            lv = float(loss.detach().float().cpu())
            pbar.set_postfix(loss=f"{lv:.4f}")
            if step % args.log_every == 0:
                print(f"step={step} loss={lv:.4f}", flush=True)
        save_projector(model, Path(args.out))
        print(f"saved projector -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
