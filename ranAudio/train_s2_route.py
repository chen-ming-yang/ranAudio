#!/usr/bin/env python3
"""S2: LoRA FastLM on <mode:speak|route>; encoder frozen; <asr> down-weighted."""

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

from ranAudio.build_route_samples import route_target
from ranAudio.model import MiniCPM5Audio, save_projector
from ranAudio.paths import MINICPM5, MINICPMO, ROUTE_JSONL, RUNS_DIR, SENSEVOICE, resolve_wav
from ranAudio.protocol import ROUTE_SYSTEM, assistant_token_weights, weighted_lm_loss


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def attach_weights(model: MiniCPM5Audio, batch: dict, assistant: str, asr_weight: float) -> torch.Tensor:
    w = torch.zeros_like(batch["labels"], dtype=torch.float32)
    aw = assistant_token_weights(
        model.tokenizer, assistant, asr_weight=asr_weight, mode_weight=1.0, body_weight=1.0
    )
    prefix = int(batch["prefix_len"])
    n = min(len(aw), int(batch["labels"].shape[1]) - prefix)
    if n > 0:
        w[0, prefix : prefix + n] = torch.tensor(aw[:n], dtype=torch.float32)
    return w.to(device=batch["labels"].device)


def train(args: argparse.Namespace) -> int:
    raw = load_jsonl(Path(args.samples))
    for row in raw:
        if not row.get("target"):
            row["target"] = route_target(row)
    pos = [r for r in raw if r.get("need_fc") or r.get("mode") == "route"]
    neg = [r for r in raw if not (r.get("need_fc") or r.get("mode") == "route")]
    if args.phase == "route":
        samples = list(pos)
    elif args.phase == "speak":
        samples = list(neg)
    else:
        samples = list(pos) + list(neg) * max(1, args.neg_repeat)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(
        f"s2 phase={args.phase} samples={len(samples)} pos={len(pos)} neg={len(neg)} "
        f"asr_weight={args.asr_weight}",
        flush=True,
    )

    proj = Path(args.projector) if Path(args.projector).is_file() else None
    enc = Path(args.encoder) if args.encoder and Path(args.encoder).is_file() else None
    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=proj,
        encoder_path=enc,
        encoder_backend=args.encoder_backend,
        sensevoice_path=args.sensevoice or SENSEVOICE,
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
    print(f"trainable={sum(p.numel() for p in trainable):,}", flush=True)
    model.train()
    opt = AdamW(trainable, lr=args.lr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    step = 0
    max_steps = args.max_steps or 0
    for epoch in range(args.epochs):
        order = list(samples)
        rng.shuffle(order)
        pbar = tqdm(order, desc=f"s2 {args.phase} {epoch+1}/{args.epochs}")
        for row in pbar:
            wav, _ = librosa.load(str(resolve_wav(row["wav"])), sr=16000, mono=True)
            assistant = str(row["target"])
            batch = model.teacher_force_batch(
                wav, assistant=assistant, tools=None, system=ROUTE_SYSTEM
            )
            if batch["n_label_tokens"] < 2:
                continue
            tw = attach_weights(model, batch, assistant, args.asr_weight)
            opt.zero_grad(set_to_none=True)
            out = model.llm(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
            )
            loss = weighted_lm_loss(out.logits, batch["labels"], tw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            lv = float(loss.detach().float().cpu())
            pbar.set_postfix(loss=f"{lv:.4f}")
            if args.log_every and step % args.log_every == 0:
                print(f"step={step} loss={lv:.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        model.llm.save_pretrained(str(out_dir / f"adapter_epoch{epoch+1}"))
        save_projector(model, out_dir / f"projector_epoch{epoch+1}.pt")
        if max_steps and step >= max_steps:
            break
    model.llm.save_pretrained(str(out_dir / "adapter_final"))
    save_projector(model, out_dir / "projector_final.pt")
    print(f"done steps={step} -> {out_dir}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROUTE_JSONL))
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "s15_asr" / "projector.pt"))
    ap.add_argument("--encoder", default=str(RUNS_DIR / "s15_asr" / "encoder.pt"))
    ap.add_argument("--encoder-backend", default="sensevoice")
    ap.add_argument("--sensevoice", default="")
    ap.add_argument("--out-dir", default=str(RUNS_DIR / "s2_route"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--asr-weight", type=float, default=0.2)
    ap.add_argument("--phase", choices=("mix", "route", "speak"), default="mix")
    ap.add_argument("--neg-repeat", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> int:
    return train(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
