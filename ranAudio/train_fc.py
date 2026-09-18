#!/usr/bin/env python3
"""LoRA MiniCPM5 + train projector: speech -> car tool XML."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import librosa
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio, save_projector
from ranAudio.paths import DATA_DIR, MINICPM5, MINICPMO, RUNS_DIR, resolve_wav
from ranAudio.tools import format_tool_xml


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
    ap.add_argument("--projector", default=str(RUNS_DIR / "align" / "projector.pt"))
    ap.add_argument("--out-dir", default=str(RUNS_DIR / "fc"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = load_jsonl(Path(args.samples))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"fc samples={len(samples)}", flush=True)

    proj = Path(args.projector) if Path(args.projector).is_file() else None
    model = MiniCPM5Audio.from_pretrained(
        args.llm, args.apm, projector_path=proj, encoder_backend="whisper"
    )
    for p in model.apm.parameters():
        p.requires_grad = False
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    for p in model.projector.parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable={n_train:,}", flush=True)
    model.train()
    opt = AdamW(trainable, lr=args.lr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    step = 0
    for epoch in range(args.epochs):
        order = list(samples)
        rng.shuffle(order)
        pbar = tqdm(order, desc=f"fc {epoch+1}/{args.epochs}")
        for row in pbar:
            wav, _ = librosa.load(str(resolve_wav(row["wav"])), sr=16000, mono=True)
            assistant = format_tool_xml(row["tool"], row.get("arguments") or {})
            batch = model.teacher_force_batch(wav, assistant=assistant, tools=row.get("tools"))
            if batch["n_label_tokens"] < 4:
                print(f"warn few labels {row.get('id')} n={batch['n_label_tokens']}", flush=True)
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
        adapter = out_dir / f"adapter_epoch{epoch+1}"
        model.llm.save_pretrained(str(adapter))
        save_projector(model, out_dir / f"projector_epoch{epoch+1}.pt")
        print(f"saved {adapter}", flush=True)
    model.llm.save_pretrained(str(out_dir / "adapter_final"))
    save_projector(model, out_dir / "projector_final.pt")
    print(f"done steps={step} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
