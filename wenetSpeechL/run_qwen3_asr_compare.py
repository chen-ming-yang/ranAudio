#!/usr/bin/env python3
"""Run Qwen3-ASR on WenetSpeech L sample manifest; write shard JSONL with CER vs official text."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Prefer installed qwen-asr (editable); do not force /cmy/cmy/Qwen3-ASR onto path
# (that tree may disagree with the env's transformers pin).

import torch
from qwen_asr import Qwen3ASRModel

from audio_slice import cer_chars, load_slice


def load_done_ids(out_dir: Path) -> Set[str]:
    done: Set[str] = set()
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                # Failed slices (e.g. ffmpeg missing) must be retried, not treated as done.
                if row.get("error"):
                    continue
                utt_id = row.get("utt_id")
                if utt_id:
                    done.add(utt_id)
    return done


def iter_manifest(path: Path, max_samples: int = 0):
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if max_samples > 0 and n >= max_samples:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/sample_manifest.jsonl"),
    )
    parser.add_argument("--model", type=str, default="/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument(
        "--prefetch_workers",
        type=int,
        default=4,
        help="CPU threads to load next audio batch while GPU transcribes",
    )
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=50000)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/qwen3_asr_L10pct"),
    )
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.out_dir)
    print(f"[qwen] resume: already done={len(done)}", flush=True)

    # Stream manifest: do not preload all remaining into RAM (needed for ~90% / 10M+ utts)
    def pending_batches():
        batch: List[Dict[str, Any]] = []
        seen = 0
        for item in iter_manifest(args.manifest, args.max_samples):
            if item["utt_id"] in done:
                continue
            batch.append(item)
            seen += 1
            if len(batch) >= args.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        print(f"[qwen] pending_streamed≈{seen} (skipped done={len(done)})", flush=True)

    # Peek whether anything remains without building full list
    has_any = False
    for item in iter_manifest(args.manifest, args.max_samples):
        if item["utt_id"] not in done:
            has_any = True
            break
    if not has_any:
        print("[qwen] nothing to do", flush=True)
        return

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    print(
        f"[qwen] loading model {args.model} batch_size={args.batch_size} "
        f"prefetch={args.prefetch_workers} ...",
        flush=True,
    )
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device,
        max_inference_batch_size=args.batch_size,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # continue shard index
    existing = list(args.out_dir.glob("shard_*.jsonl"))
    shard_idx = len(existing)
    shard_path = args.out_dir / f"shard_{shard_idx:04d}.jsonl"
    shard_f = shard_path.open("a", encoding="utf-8")
    in_shard = 0

    total_err = 0
    total_ref = 0
    ok = 0
    fail = 0
    t0 = time.perf_counter()

    def _load_batch(
        batch: List[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        audios: List[Any] = []
        meta: List[Dict[str, Any]] = []
        load_fails: List[Dict[str, Any]] = []
        for it in batch:
            try:
                wav, sr = load_slice(it["audio_path"], it["start"], it["duration"])
                audios.append((wav, sr))
                meta.append(it)
            except Exception as exc:
                load_fails.append(
                    {
                        "utt_id": it["utt_id"],
                        "ref": it["text"],
                        "hyp": "",
                        "cer": 1.0,
                        "confidence": it.get("confidence"),
                        "source": it.get("source"),
                        "model": "qwen3-asr-1.7b",
                        "error": str(exc)[:300],
                    }
                )
        return audios, meta, load_fails

    try:
        batch_iter = pending_batches()
        workers = max(0, int(args.prefetch_workers))
        pool = ThreadPoolExecutor(max_workers=max(1, workers)) if workers > 0 else None
        try:
            cur = next(batch_iter, None)
            fut = pool.submit(_load_batch, cur) if pool and cur is not None else None
            while cur is not None:
                nxt = next(batch_iter, None)
                nxt_fut = pool.submit(_load_batch, nxt) if pool and nxt is not None else None
                if fut is not None:
                    audios, meta, load_fails = fut.result()
                else:
                    audios, meta, load_fails = _load_batch(cur)

                for err_row in load_fails:
                    fail += 1
                    shard_f.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    in_shard += 1

                if audios:
                    with torch.inference_mode():
                        results = model.transcribe(
                            audio=audios,
                            language=args.language if args.language else None,
                        )

                    for it, r in zip(meta, results):
                        hyp = (r.text or "").strip() if r is not None else ""
                        cer, errs, reflen = cer_chars(it["text"], hyp)
                        total_err += errs
                        total_ref += reflen
                        ok += 1
                        out = {
                            "utt_id": it["utt_id"],
                            "ref": it["text"],
                            "hyp": hyp,
                            "cer": round(cer, 6),
                            "confidence": it.get("confidence"),
                            "source": it.get("source"),
                            "model": "qwen3-asr-1.7b",
                            "start": it["start"],
                            "duration": it["duration"],
                            "audio_path": it["audio_path"],
                        }
                        shard_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        in_shard += 1

                if in_shard >= args.shard_size:
                    shard_f.flush()
                    shard_f.close()
                    shard_idx += 1
                    shard_path = args.out_dir / f"shard_{shard_idx:04d}.jsonl"
                    shard_f = shard_path.open("a", encoding="utf-8")
                    in_shard = 0

                done_n = ok + fail
                if done_n > 0 and done_n % args.log_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate = done_n / max(elapsed, 1e-6)
                    gcer = total_err / total_ref if total_ref else 0.0
                    print(
                        f"[qwen] done={done_n} (+prev {len(done)}) ok={ok} fail={fail} "
                        f"gCER={gcer:.4f} {rate:.2f} utt/s elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )
                    shard_f.flush()

                cur, fut = nxt, nxt_fut
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

        # final log
        done_n = ok + fail
        elapsed = time.perf_counter() - t0
        rate = done_n / max(elapsed, 1e-6)
        gcer = total_err / total_ref if total_ref else 0.0
        print(
            f"[qwen] done={done_n} (+prev {len(done)}) ok={ok} fail={fail} "
            f"gCER={gcer:.4f} {rate:.2f} utt/s elapsed={elapsed/60:.1f}m",
            flush=True,
        )
    finally:
        shard_f.flush()
        shard_f.close()

    gcer = total_err / total_ref if total_ref else 0.0
    summary = {
        "model": "qwen3-asr-1.7b",
        "ok": ok,
        "fail": fail,
        "already_done_before": len(done),
        "global_cer": gcer,
        "total_errors": total_err,
        "total_ref_chars": total_ref,
        "elapsed_sec": time.perf_counter() - t0,
        "out_dir": str(args.out_dir),
    }
    with (args.out_dir / "last_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
