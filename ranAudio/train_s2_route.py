#!/usr/bin/env python3
"""S2 LoRA: <|FC|><|BASE|> / <|NOFC|>. Encoder+projector frozen. Four-phase resume."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import librosa
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, S2_AUDIO_JSONL, SENSEVOICE, resolve_wav
from ranAudio.protocol import S2_SYSTEM, assistant_token_weights, weighted_lm_loss
from transformers import AutoTokenizer
from ranAudio.s2_catalog import read_jsonl

S15_ENC = RUNS_DIR / "s15_asr" / "b_ms_cont" / "encoder_best.pt"
S15_PROJ = RUNS_DIR / "s15_asr" / "b_ms_cont" / "projector_best.pt"
PHASES = ("route", "chat", "asr", "mix", "speak")


def _bucket(row: dict) -> str:
    if row.get("route") == "fc" or row.get("need_fc") or row.get("mode") == "route":
        return "fc"
    if row.get("kind") == "asr":
        return "asr"
    return "chat"


def select_samples(raw: list[dict], phase: str, neg_repeat: int) -> list[dict]:
    fc = [r for r in raw if _bucket(r) == "fc"]
    chat = [r for r in raw if _bucket(r) == "chat"]
    asr = [r for r in raw if _bucket(r) == "asr"]
    if phase == "route":
        return list(fc)
    if phase in ("chat", "speak"):
        return list(chat)
    if phase == "asr":
        return list(asr)
    return list(fc) + list(chat) * max(1, neg_repeat) + list(asr)


def attach_weights(model: MiniCPM5Audio, batch: dict, assistant: str, asr_weight: float) -> torch.Tensor:
    w = torch.zeros_like(batch["labels"], dtype=torch.float32)
    aw = assistant_token_weights(
        model.tokenizer, assistant, asr_weight=asr_weight, mode_weight=1.0, body_weight=1.0
    )
    labels = batch["labels"]
    prefix = int(batch["prefix_len"])
    n = min(len(aw), int(labels.shape[1]) - prefix)
    if n > 0:
        w[0, prefix : prefix + n] = torch.tensor(aw[:n], dtype=torch.float32)
    # <|im_end|> sits after the assistant string and must receive loss.
    tail = labels[0, prefix + n :]
    if tail.numel():
        supervised = tail != -100
        if bool(supervised.any()):
            idx = torch.arange(prefix + n, int(labels.shape[1]), device=labels.device)
            w[0, idx[supervised]] = 1.0
    return w.to(device=labels.device)


def _load_adapter(model: MiniCPM5Audio, resume: str | None, lora_cfg: LoraConfig) -> None:
    if resume:
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(f"resume adapter missing: {path}")
        try:
            model.llm = PeftModel.from_pretrained(model.llm, str(path), is_trainable=True)
        except TypeError:
            model.llm = PeftModel.from_pretrained(model.llm, str(path))
            for name, p in model.llm.named_parameters():
                p.requires_grad = "lora_" in name
        print(f"resume adapter {path}", flush=True)
        return
    model.llm = get_peft_model(model.llm, lora_cfg)


def train(args: argparse.Namespace) -> int:
    raw = read_jsonl(Path(args.samples))
    raw = [r for r in raw if r.get("target") and r.get("wav")]
    samples = select_samples(raw, args.phase, args.neg_repeat)
    if args.max_samples:
        samples = samples[: args.max_samples]
    n_fc = sum(1 for r in raw if _bucket(r) == "fc")
    n_chat = sum(1 for r in raw if _bucket(r) == "chat")
    n_asr = sum(1 for r in raw if _bucket(r) == "asr")
    print(
        f"s2 phase={args.phase} samples={len(samples)} "
        f"pool fc={n_fc} chat={n_chat} asr={n_asr} asr_weight={args.asr_weight}",
        flush=True,
    )
    if not samples:
        print("no samples for this phase", flush=True)
        return 1

    proj = Path(args.projector) if Path(args.projector).is_file() else None
    enc = Path(args.encoder) if args.encoder and Path(args.encoder).is_file() else None
    tokenizer = None
    if args.resume:
        tok_dir = Path(args.resume) / "tokenizer"
        if tok_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
            print(f"reuse tokenizer from {tok_dir}", flush=True)
    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=proj,
        encoder_path=enc,
        encoder_backend=args.encoder_backend,
        sensevoice_path=args.sensevoice or SENSEVOICE,
        tokenizer=tokenizer,
    )
    for p in model.apm.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = False
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    _load_adapter(model, args.resume, lora_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable={sum(p.numel() for p in trainable):,}", flush=True)
    model.train()
    opt = AdamW(trainable, lr=args.lr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if proj and proj.is_file():
        dst = out_dir / "projector_final.pt"
        if not dst.exists():
            shutil.copy2(proj, dst)

    rng = random.Random(args.seed)
    step = 0
    max_steps = args.max_steps or 0
    for epoch in range(args.epochs):
        order = list(samples)
        rng.shuffle(order)
        pbar = tqdm(order, desc=f"s2 {args.phase} {epoch+1}/{args.epochs}")
        for row in pbar:
            wav_path = resolve_wav(row["wav"])
            if not wav_path.is_file():
                continue
            try:
                wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            except Exception as exc:  # noqa: BLE001
                print(f"skip wav {row.get('id')}: {exc}", flush=True)
                continue
            assistant = str(row["target"])
            batch = model.teacher_force_batch(
                wav, assistant=assistant, tools=None, system=S2_SYSTEM
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
        if max_steps and step >= max_steps:
            break
    model.llm.save_pretrained(str(out_dir / "adapter_final"))
    model.tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    print(f"done steps={step} -> {out_dir}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(S2_AUDIO_JSONL))
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(S15_PROJ))
    ap.add_argument("--encoder", default=str(S15_ENC))
    ap.add_argument("--encoder-backend", default="sensevoice")
    ap.add_argument("--sensevoice", default="")
    ap.add_argument("--out-dir", default=str(RUNS_DIR / "s2_fc_nofc" / "mix"))
    ap.add_argument("--resume", default="", help="previous phase adapter dir")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--asr-weight", type=float, default=0.2)
    ap.add_argument("--phase", choices=PHASES, default="mix")
    ap.add_argument("--neg-repeat", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    args.resume = args.resume or None
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
