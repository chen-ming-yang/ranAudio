#!/usr/bin/env python3
"""Evaluate WhisperX baseline and/or fine-tuned aligner on gold holdout.

Examples:
  python char_align/eval_align.py --split test --baseline whisperx
  PYTHONPATH=/cmy/cmy/real-asr:$PWD python char_align/eval_align.py \\
      --split test --checkpoint runs/char_align/s06_part1/best.pt
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

from char_align.dataset import CharAlignDataset, char_token_ids, load_jsonl  # noqa: E402
from char_align.metrics import AlignStats  # noqa: E402
from char_align.model_utils import (  # noqa: E402
    forced_align_onsets,
    load_asr_bundle,
    onset_monotonic_dp,
    onset_peaks_onsets,
)


def eval_whisperx(rows: list[dict], wx_dir: Path) -> dict:
    stats = AlignStats()
    missing = 0
    for r in rows:
        wp = wx_dir / f"{r['id']}.json"
        if not wp.is_file():
            missing += 1
            stats.n_utt += 1
            stats.n_skip += 1
            continue
        wx = json.loads(wp.read_text(encoding="utf-8"))
        gold = [c["onset"] for c in r["chars"]]
        if not isinstance(wx, list) or len(wx) != len(gold):
            stats.n_utt += 1
            stats.n_skip += 1
            continue
        pred = [float(x["start"]) for x in wx]
        # optional char check
        if any(x.get("char") != c["char"] for x, c in zip(wx, r["chars"])):
            stats.n_utt += 1
            stats.n_skip += 1
            continue
        stats.add_utt(gold, pred)
    out = stats.summary()
    out["missing_wx"] = missing
    out["method"] = "whisperx"
    return out


@torch.no_grad()
def eval_model(
    rows: list[dict],
    checkpoint: str,
    tokenizer: str,
    config: str,
    base_ckpt: str,
    device: str,
    mode: str = "onset",
    decode: str = "dp",
) -> dict:
    cfg, transducer, ctc_head, onset_head = load_asr_bundle(
        config=config, checkpoint=base_ckpt, device=device
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    transducer.load_state_dict(ckpt["model"], strict=False)
    ctc_head.load_state_dict(ckpt["ctc_head"], strict=False)
    if "onset_head" in ckpt:
        onset_head.load_state_dict(ckpt["onset_head"], strict=False)
    frame_shift_s = float(ckpt.get("frame_shift_s") or 0.08)
    transducer.eval()
    ctc_head.eval()
    onset_head.eval()

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer)
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80
    ).to(device)

    stats = AlignStats()
    for r in rows:
        audio, sr = sf.read(r["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        wav = torch.from_numpy(np.ascontiguousarray(audio))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.to(device)
        feat = fbank(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
        feat = torch.log(feat.clamp(min=1e-5)).unsqueeze(0)
        feat_lens = torch.tensor([feat.size(1)], device=device)
        enc, enc_lens = transducer.encoder(feat, feat_lens)
        gold = [c["onset"] for c in r["chars"]]
        n_chars = len(gold)
        if mode == "onset":
            onset_logits = onset_head(enc)[0]  # (T,)
            if decode == "dp":
                pred = onset_monotonic_dp(
                    onset_logits, n_chars, int(enc_lens[0].item()), frame_shift_s
                )
            else:
                pred = onset_peaks_onsets(
                    onset_logits, n_chars, int(enc_lens[0].item()), frame_shift_s
                )
        else:  # fa
            log_probs = ctc_head(enc)[0]  # (T,V)
            tids = torch.tensor(char_token_ids(sp, r["text"]), device=device)
            pred = forced_align_onsets(
                log_probs, tids, int(enc_lens[0].item()), frame_shift_s, blank_id=0
            )
        stats.add_utt(gold, pred)

    out = stats.summary()
    out["method"] = "onset_head" if mode == "onset" else "zipformer_fa"
    out["checkpoint"] = checkpoint
    out["frame_shift_s"] = frame_shift_s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/char_align/s06_part1"))
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--baseline", choices=["whisperx", "none"], default="none")
    ap.add_argument(
        "--wx-dir",
        type=Path,
        default=Path("/cmy/cmy/DARS/data/mdsd/whisperx_align"),
    )
    ap.add_argument("--checkpoint", default="")
    ap.add_argument(
        "--method",
        default="onset",
        choices=["onset", "fa"],
        help="onset=head peak-picking; fa=CTC forced-align",
    )
    ap.add_argument(
        "--decode",
        default="dp",
        choices=["dp", "topn"],
        help="onset decode: dp=monotonic DP, topn=global top-N peaks",
    )
    ap.add_argument(
        "--eval-base",
        action="store_true",
        help="Evaluate dysarthric CTC forced-align without align fine-tune",
    )
    ap.add_argument("--tokenizer", default="/cmy/cmy/real-asr/data/tokenizer/sp_6000.model")
    ap.add_argument("--config", default="/cmy/cmy/real-asr/configs/train_finetune_dysarthric.yaml")
    ap.add_argument(
        "--base-checkpoint",
        default="/cmy/cmy/real-asr/exp/finetune_dysarthric/epoch_30.pt",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = load_jsonl(args.data_dir / f"{args.split}.jsonl")
    results = {"split": args.split, "n": len(rows)}

    if args.baseline == "whisperx":
        results["whisperx"] = eval_whisperx(rows, args.wx_dir)
        print(json.dumps(results["whisperx"], ensure_ascii=False, indent=2))

    if args.eval_base:
        # synthesize a ckpt that only carries base ASR weights
        cfg, transducer, ctc_head, onset_head = load_asr_bundle(
            config=args.config, checkpoint=args.base_checkpoint, device="cpu"
        )
        tmp = Path("tmp") / "char_align_base_fa.pt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": transducer.state_dict(),
                "ctc_head": ctc_head.state_dict(),
                "onset_head": onset_head.state_dict(),
                "frame_shift_s": 0.08,
            },
            tmp,
        )
        results["base_fa"] = eval_model(
            rows,
            checkpoint=str(tmp),
            tokenizer=args.tokenizer,
            config=args.config,
            base_ckpt=args.base_checkpoint,
            device=args.device,
            mode="fa",
        )
        print(json.dumps(results["base_fa"], ensure_ascii=False, indent=2))

    if args.checkpoint:
        results["model"] = eval_model(
            rows,
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            config=args.config,
            base_ckpt=args.base_checkpoint,
            device=args.device,
            mode=args.method,
            decode=args.decode,
        )
        print(json.dumps(results["model"], ensure_ascii=False, indent=2))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
