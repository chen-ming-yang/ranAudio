#!/usr/bin/env python3
"""S1.5: freeze MiniCPM5-2B, train SenseVoice encoder + new 512→2048 projector."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import (
    MiniCPM5Audio,
    freeze_except_projector,
    freeze_llm_train_encoder,
    save_encoder,
    save_projector,
)
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, SENSEVOICE, WENET_S_JSONL
from ranAudio.protocol import ASR_SYSTEM, cer
from ranAudio.wenet import (
    CompactWavDataset,
    DurationBatchSampler,
    IndexedCompact,
    Segment,
    collate_wav_batch,
    default_split_paths,
    export_s_default,
    iter_supervisions,
    load_recordings,
    load_segment_wav,
)


class _StdoutTee:
    """Duplicate stdout to a file. tqdm stays on stderr so the log stays line-based."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fp = path.open("a", encoding="utf-8")
        self._console = sys.__stdout__
        self._fp.write(f"\n# {datetime.now().isoformat(timespec='seconds')} logging -> {path}\n")
        self._fp.flush()

    def write(self, s: str) -> int:
        self._console.write(s)
        self._fp.write(s)
        if "\n" in s:
            self._fp.flush()
            self._console.flush()
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        self._fp.flush()

    def isatty(self) -> bool:
        return False


def _start_train_log(out_dir: Path) -> Path:
    log_path = out_dir / "train.log"
    sys.stdout = _StdoutTee(log_path)
    print(f"train log {log_path}", flush=True)
    return log_path


def _llm_trainable(model: MiniCPM5Audio) -> int:
    return sum(int(p.requires_grad) for p in model.llm.parameters())


def _load_eval_segments(split: str, max_n: int) -> list[Segment]:
    sup, rec = default_split_paths(split)
    if not sup.is_file() or not rec.is_file():
        return []
    recs = load_recordings(rec)
    segs: list[Segment] = []
    for seg in iter_supervisions(sup, recordings=recs, require_audio=True):
        segs.append(seg)
        if max_n and len(segs) >= max_n:
            break
    return segs


@torch.no_grad()
def _eval_cer(
    model: MiniCPM5Audio,
    segs: list[Segment],
    *,
    log_path: Path | None = None,
    step: int | None = None,
) -> float | None:
    if not segs:
        return None
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    rows: list[str] = []
    for seg in segs:
        try:
            wav = load_segment_wav(seg)
        except Exception as exc:
            print(f"eval skip {seg.sid}: {exc}", flush=True)
            continue
        if wav.size < 1600:
            continue
        hyp = model.generate_text(wav, tools=None, system=ASR_SYSTEM, max_new_tokens=64)
        score = cer(seg.text, hyp)
        total += score
        n += 1
        rows.append(f"cer={score:.4f} id={seg.sid} ref={seg.text!r} hyp={hyp!r}")
    mean = (total / n) if n else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        header = [f"step={step} n={n} cer={mean}"]
        log_path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
        print(f"eval log {log_path} n={n} cer={mean}", flush=True)
    if was_training:
        model.train()
        model.llm.eval()
    return mean


def _restore_train_mode(model: MiniCPM5Audio, freeze_encoder: bool) -> None:
    model.train()
    model.llm.eval()
    if freeze_encoder:
        model.apm.eval()


def _step_ckpt_paths(out_dir: Path, step: int, *, freeze_encoder: bool) -> list[Path]:
    paths = [out_dir / f"projector_step{step}.pt"]
    if not freeze_encoder:
        paths.append(out_dir / f"encoder_step{step}.pt")
    return paths


def _link_best(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.hardlink_to(src)


def _prune_step_ckpts(
    out_dir: Path,
    scored: list[tuple[float, int]],
    keep: int,
    *,
    freeze_encoder: bool,
) -> None:
    """Keep the `keep` lowest-CER step checkpoints; delete the rest."""
    if keep <= 0 or not scored:
        return
    ranked = sorted(scored, key=lambda x: (x[0], -x[1]))
    keep_steps = {step for _, step in ranked[:keep]}
    for path in out_dir.glob("projector_step*.pt"):
        try:
            step = int(path.stem.removeprefix("projector_step"))
        except ValueError:
            continue
        if step in keep_steps:
            continue
        for victim in _step_ckpt_paths(out_dir, step, freeze_encoder=freeze_encoder):
            if victim.is_file():
                victim.unlink()
                print(f"pruned {victim.name}", flush=True)
    (out_dir / "ckpts.json").write_text(
        json.dumps(
            {
                "keep": keep,
                "kept": [{"step": step, "cer": cer} for cer, step in ranked[:keep]],
            },
            indent=2,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(WENET_S_JSONL), help="compact jsonl from prepare_wenet")
    ap.add_argument("--prepare-s", action="store_true", help="export S subset if --samples missing")
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default="", help="optional prior 512-d projector; empty = random init")
    ap.add_argument("--encoder-backend", default="sensevoice")
    ap.add_argument("--sensevoice", default="")
    ap.add_argument("--encoder", default="", help="optional previous S1.5 encoder")
    ap.add_argument("--out-dir", default=str(RUNS_DIR / "s15_asr"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4, help="projector lr")
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--freeze-encoder", action="store_true", help="S1.5a: train projector only")
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=16.0)
    ap.add_argument("--accum", type=int, default=1, help="optimizer step every N batches")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--max-batch-dur", type=float, default=32.0, help="seconds of audio per GPU batch")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--keep-best", type=int, default=5, help="keep this many lowest-CER step ckpts")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-split", default="aishell-test")
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _start_train_log(out_dir)
    (out_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))

    samples_path = Path(args.samples)
    if not samples_path.is_file():
        if args.prepare_s:
            print(f"exporting S subset -> {samples_path}", flush=True)
            n = export_s_default(samples_path, max_samples=args.max_samples)
            print(f"exported {n}", flush=True)
        else:
            print(f"missing {samples_path}; run prepare_wenet or pass --prepare-s", flush=True)
            return 1

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    indexed = IndexedCompact(samples_path)
    order = list(range(len(indexed)))
    if args.max_samples:
        order = order[: args.max_samples]
    dataset = CompactWavDataset(indexed, order, min_dur=args.min_dur, max_dur=args.max_dur)
    sampler = DurationBatchSampler(
        dataset.durations,
        args.max_batch_dur,
        shuffle=True,
        seed=args.seed,
    )
    print(
        f"s15 samples={len(dataset)} of {len(indexed)} batches~{len(sampler)} "
        f"max_batch_dur={args.max_batch_dur} workers={args.num_workers}",
        flush=True,
    )

    proj = Path(args.projector) if args.projector and Path(args.projector).is_file() else None
    enc = Path(args.encoder) if args.encoder and Path(args.encoder).is_file() else None
    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=proj,
        encoder_path=enc,
        encoder_backend=args.encoder_backend,
        sensevoice_path=args.sensevoice or SENSEVOICE,
    )
    if args.freeze_encoder:
        trainable = freeze_except_projector(model)
        opt = AdamW(trainable, lr=args.lr)
    else:
        trainable = freeze_llm_train_encoder(model)
        opt = AdamW(
            [
                {"params": list(model.apm.parameters()), "lr": args.encoder_lr},
                {"params": list(model.projector.parameters()), "lr": args.lr},
            ]
        )
    n_llm = _llm_trainable(model)
    print(
        f"trainable={sum(p.numel() for p in trainable):,} llm_trainable_tensors={n_llm} "
        f"freeze_encoder={args.freeze_encoder} proj_lr={args.lr} enc_lr={args.encoder_lr}",
        flush=True,
    )
    if n_llm:
        print("refusing to train: 2B has requires_grad=True", flush=True)
        return 1
    _restore_train_mode(model, args.freeze_encoder)

    eval_segs = _load_eval_segments(args.eval_split, args.eval_n)
    print(f"eval {args.eval_split} n={len(eval_segs)}", flush=True)

    step = 0
    opt_steps = 0
    accum = max(1, int(args.accum))
    running = 0.0
    running_n = 0
    best_cer: float | None = None
    scored_ckpts: list[tuple[float, int]] = []
    next_eval = args.eval_every
    next_save = args.save_every
    opt.zero_grad(set_to_none=True)

    dl_kwargs: dict = {
        "batch_sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate_wav_batch,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = max(2, int(args.prefetch))
    loader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

    def _maybe_eval_and_save(*, force: bool = False) -> None:
        nonlocal best_cer, next_eval, next_save
        do_eval = force or (args.eval_every and step >= next_eval)
        do_save = force or (args.save_every and step >= next_save)
        if not do_eval and not do_save:
            return
        if do_save:
            save_projector(model, out_dir / f"projector_step{step}.pt")
            if not args.freeze_encoder:
                save_encoder(model, out_dir / f"encoder_step{step}.pt")
            print(f"saved step={step} -> {out_dir}", flush=True)
            if args.save_every:
                next_save += args.save_every
        if do_eval and eval_segs:
            score = _eval_cer(
                model,
                eval_segs,
                log_path=out_dir / f"eval_step{step}.log",
                step=step,
            )
            _restore_train_mode(model, args.freeze_encoder)
            print(f"eval step={step} cer={score}", flush=True)
            if score is not None:
                scored_ckpts.append((float(score), step))
                if best_cer is None or score < best_cer:
                    best_cer = score
                    proj_step = out_dir / f"projector_step{step}.pt"
                    if proj_step.is_file():
                        _link_best(proj_step, out_dir / "projector_best.pt")
                    else:
                        save_projector(model, out_dir / "projector_best.pt")
                    if not args.freeze_encoder:
                        enc_step = out_dir / f"encoder_step{step}.pt"
                        if enc_step.is_file():
                            _link_best(enc_step, out_dir / "encoder_best.pt")
                        else:
                            save_encoder(model, out_dir / "encoder_best.pt")
                    (out_dir / "best.json").write_text(
                        json.dumps({"step": step, "cer": best_cer, "split": args.eval_split}, indent=2)
                    )
                    print(f"best cer={best_cer:.4f} step={step}", flush=True)
            if args.eval_every:
                next_eval += args.eval_every
        if do_save:
            _prune_step_ckpts(
                out_dir,
                scored_ckpts,
                args.keep_best,
                freeze_encoder=args.freeze_encoder,
            )

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"s15 {epoch+1}/{args.epochs}")
        for raw in pbar:
            if not raw:
                continue
            batch = model.teacher_force_many(
                raw["wavs"], raw["texts"], tools=None, system=ASR_SYSTEM
            )
            if batch["n_label_tokens"] < 1:
                continue
            out = model.llm(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / accum
            loss.backward()
            running += float(loss.detach().float().cpu()) * accum
            running_n += 1
            n_utt = int(batch.get("batch_size") or len(raw["wavs"]))
            step += n_utt
            if running_n % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_steps += 1
            lv = running / max(1, running_n)
            dur_sum = float(sum(raw["durations"]))
            pbar.set_postfix(loss=f"{lv:.4f}", step=step, bs=n_utt)
            if step <= n_utt or step % args.log_every < n_utt:
                print(
                    f"step={step} opt_steps={opt_steps} loss={lv:.4f} "
                    f"bs={n_utt} batch_dur={dur_sum:.1f}",
                    flush=True,
                )
                running = 0.0
                running_n = 0
            if opt_steps == 1 and n_utt:
                bad = [n for n, p in model.llm.named_parameters() if p.grad is not None]
                print(f"llm_grad_tensors={len(bad)} (want 0)", flush=True)
            _maybe_eval_and_save()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if step % accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    _maybe_eval_and_save(force=True)
    print(f"done steps={step} opt_steps={opt_steps} best_cer={best_cer} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
