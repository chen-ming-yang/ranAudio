#!/usr/bin/env python3
"""S2 full fine-tune: <|FC|><|BASE|> / <|NOFC|>. Encoder+projector frozen."""

from __future__ import annotations

import argparse
import random
import shutil
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

from ranAudio.model import MiniCPM5Audio, save_projector
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, S2_AUDIO_JSONL, SENSEVOICE, resolve_wav
from ranAudio.protocol import S2_SYSTEM, assistant_token_weights, weighted_lm_loss
from ranAudio.s2_catalog import read_jsonl
from transformers import AutoTokenizer

S15_ENC = RUNS_DIR / "s15_asr" / "b_ms_cont" / "encoder_best.pt"
S15_PROJ = RUNS_DIR / "s15_asr" / "b_ms_cont" / "projector_best.pt"
PHASES = ("route", "chat", "asr", "mix", "speak")


def _bucket(row: dict) -> str:
    if row.get("route") == "fc" or row.get("need_fc") or row.get("mode") == "route":
        return "fc"
    if row.get("kind") == "asr":
        return "asr"
    return "chat"


def select_samples(
    raw: list[dict], phase: str, neg_repeat: int, boundary_repeat: int = 1
) -> list[dict]:
    fc = [r for r in raw if _bucket(r) == "fc"]
    chat = [r for r in raw if _bucket(r) == "chat"]
    asr = [r for r in raw if _bucket(r) == "asr"]
    if phase == "route":
        mixed = list(fc)
    elif phase in ("chat", "speak"):
        mixed = list(chat)
    elif phase == "asr":
        mixed = list(asr)
    else:
        mixed = list(fc) + list(chat) * max(1, neg_repeat) + list(asr)
    extra = max(1, boundary_repeat) - 1
    if extra <= 0:
        return mixed
    out = list(mixed)
    for row in mixed:
        if row.get("source") == "boundary":
            out.extend([row] * extra)
    return out


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
    # Chat template appends <|im_end|> after the assistant string. Those tokens
    # used to stay at weight 0, so the model never learned to stop.
    tail = labels[0, prefix + n :]
    if tail.numel():
        supervised = tail != -100
        if bool(supervised.any()):
            idx = torch.arange(prefix + n, int(labels.shape[1]), device=labels.device)
            w[0, idx[supervised]] = 1.0
    return w.to(device=labels.device)


def train(args: argparse.Namespace) -> int:
    raw = read_jsonl(Path(args.samples))
    raw = [r for r in raw if r.get("target") and r.get("wav")]
    samples = select_samples(raw, args.phase, args.neg_repeat, args.boundary_repeat)
    if args.max_samples:
        samples = samples[: args.max_samples]
    n_fc = sum(1 for r in raw if _bucket(r) == "fc")
    n_chat = sum(1 for r in raw if _bucket(r) == "chat")
    n_asr = sum(1 for r in raw if _bucket(r) == "asr")
    print(
        f"s2 full phase={args.phase} samples={len(samples)} "
        f"pool fc={n_fc} chat={n_chat} asr={n_asr} asr_weight={args.asr_weight}",
        flush=True,
    )
    if not samples:
        print("no samples for this phase", flush=True)
        return 1

    proj = Path(args.projector) if Path(args.projector).is_file() else None
    enc = Path(args.encoder) if args.encoder and Path(args.encoder).is_file() else None

    tokenizer = None
    resume_file = None
    if args.resume:
        resume_path = Path(args.resume)
        resume_dir = resume_path if resume_path.is_dir() else resume_path.parent
        resume_file = resume_path if resume_path.is_file() else resume_dir / "llm_final.safetensors"
        tok_dir = resume_dir / "tokenizer"
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
    train_projector = bool(args.train_projector)
    freeze_llm = bool(args.freeze_llm)
    for p in model.apm.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = train_projector
    if freeze_llm:
        for p in model.llm.parameters():
            p.requires_grad = False

    if resume_file is not None:
        if not resume_file.exists():
            raise FileNotFoundError(f"resume checkpoint missing: {resume_file}")
        if resume_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            sd = load_file(str(resume_file), device=str(model.device))
        else:
            sd = torch.load(resume_file, map_location=model.device, weights_only=True)
        model.llm.load_state_dict(sd)
        print(f"resume full checkpoint {resume_file}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"trainable={sum(p.numel() for p in trainable):,} "
        f"train_projector={int(train_projector)} freeze_llm={int(freeze_llm)}",
        flush=True,
    )
    if not trainable:
        print("no trainable parameters", flush=True)
        return 1
    if train_projector:
        model.llm.eval()
        model.apm.eval()
        model.projector.train()
    else:
        model.train()
    opt = AdamW(trainable, lr=args.lr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if proj and proj.is_file() and not train_projector:
        dst = out_dir / "projector_final.pt"
        if not dst.exists():
            shutil.copy2(proj, dst)

    rng = random.Random(args.seed)
    step = 0
    max_steps = args.max_steps or 0
    for epoch in range(args.epochs):
        order = list(samples)
        rng.shuffle(order)
        pbar = tqdm(order, desc=f"s2 full {args.phase} {epoch+1}/{args.epochs}")
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

        ckpt = out_dir / f"llm_epoch{epoch+1}.safetensors"
        from safetensors.torch import save_file

        save_file(model.llm.state_dict(), str(ckpt))
        model.tokenizer.save_pretrained(str(out_dir / "tokenizer"))
        if train_projector:
            save_projector(model, out_dir / "projector_final.pt")
        if max_steps and step >= max_steps:
            break

    final = out_dir / "llm_final.safetensors"
    from safetensors.torch import save_file

    save_file(model.llm.state_dict(), str(final))
    model.tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    if train_projector:
        save_projector(model, out_dir / "projector_final.pt")
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
    ap.add_argument("--out-dir", default=str(RUNS_DIR / "s2_fc_nofc_full" / "mix"))
    ap.add_argument("--resume", default="", help="previous phase checkpoint dir containing llm_final.safetensors")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--train-projector",
        action="store_true",
        help="unfreeze the projector; pair with --freeze-llm to train it alone",
    )
    ap.add_argument(
        "--freeze-llm",
        action="store_true",
        help="keep the resumed LLM fixed and still write it into the output dir",
    )
    ap.add_argument("--asr-weight", type=float, default=0.2)
    ap.add_argument("--phase", choices=PHASES, default="mix")
    ap.add_argument("--neg-repeat", type=int, default=2)
    ap.add_argument(
        "--boundary-repeat",
        type=int,
        default=1,
        help="extra copies of source=boundary rows inside the selected mix",
    )
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
