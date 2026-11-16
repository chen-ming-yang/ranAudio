#!/usr/bin/env python3
"""Run FireRedASR2-AED on WenetSpeech L sample manifest; write shard JSONL with CER."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

FIRERED_ROOT = Path("/cmy/cmy/FireRedASR2S")
if str(FIRERED_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRERED_ROOT))

from audio_slice import cer_chars, write_temp_wav


def load_done_ids(out_dir: Path) -> Set[str]:
    done: Set[str] = set()
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["utt_id"])
                except Exception:
                    continue
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
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED",
    )
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument(
        "--prefetch_workers",
        type=int,
        default=4,
        help="CPU threads to prepare next wav batch while GPU transcribes",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=50000)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/firered_asr_L10pct"),
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--tmpdir", type=str, default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.tmpdir) if args.tmpdir else args.out_dir / "tmp_wav"
    tmp_root.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(args.out_dir)
    print(f"[firered] resume: already done={len(done)}", flush=True)

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
        print(f"[firered] pending_streamed≈{seen} (skipped done={len(done)})", flush=True)

    has_any = False
    for item in iter_manifest(args.manifest, args.max_samples):
        if item["utt_id"] not in done:
            has_any = True
            break
    if not has_any:
        print("[firered] nothing to do", flush=True)
        return

    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: WPS433

    asr_config = FireRedAsr2Config(
        use_gpu=bool(args.use_gpu),
        beam_size=3,
        nbest=1,
        decode_max_len=0,
        decode_min_len=0,
        repetition_penalty=1.0,
        llm_length_penalty=0.0,
        temperature=1.0,
    )
    print(
        f"[firered] loading AED from {args.model_dir} "
        f"batch_size={args.batch_size} prefetch={args.prefetch_workers} ...",
        flush=True,
    )
    model = FireRedAsr2.from_pretrained("aed", args.model_dir, asr_config)

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

    def _prep_batch(
        batch: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        uttids: List[str] = []
        wavs: List[str] = []
        meta: List[Dict[str, Any]] = []
        tmp_files: List[str] = []
        load_fails: List[Dict[str, Any]] = []
        for it in batch:
            try:
                wav_path = write_temp_wav(
                    it["audio_path"],
                    it["start"],
                    it["duration"],
                    out_path=str(tmp_root / f"{it['utt_id']}.wav"),
                )
                tmp_files.append(wav_path)
                uttids.append(it["utt_id"])
                wavs.append(wav_path)
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
                        "model": "firered-asr2-aed",
                        "error": str(exc)[:300],
                    }
                )
        return uttids, wavs, meta, tmp_files, load_fails

    try:
        batch_iter = pending_batches()
        workers = max(0, int(args.prefetch_workers))
        pool = ThreadPoolExecutor(max_workers=max(1, workers)) if workers > 0 else None
        try:
            cur = next(batch_iter, None)
            fut = pool.submit(_prep_batch, cur) if pool and cur is not None else None
            while cur is not None:
                nxt = next(batch_iter, None)
                nxt_fut = pool.submit(_prep_batch, nxt) if pool and nxt is not None else None
                if fut is not None:
                    uttids, wavs, meta, tmp_files, load_fails = fut.result()
                else:
                    uttids, wavs, meta, tmp_files, load_fails = _prep_batch(cur)

                for err_row in load_fails:
                    fail += 1
                    shard_f.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    in_shard += 1

                if wavs:
                    try:
                        results = model.transcribe(uttids, wavs)
                        by_id = {}
                        for r in results:
                            if isinstance(r, dict):
                                by_id[r.get("uttid") or r.get("utt_id")] = r
                            else:
                                by_id[getattr(r, "uttid", None)] = r

                        for it in meta:
                            r = by_id.get(it["utt_id"])
                            if isinstance(r, dict):
                                hyp = (r.get("text") or "").strip()
                            else:
                                hyp = (getattr(r, "text", "") or "").strip()
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
                                "model": "firered-asr2-aed",
                                "start": it["start"],
                                "duration": it["duration"],
                                "audio_path": it["audio_path"],
                            }
                            shard_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                            in_shard += 1
                    except Exception as exc:
                        for it in meta:
                            fail += 1
                            err_row = {
                                "utt_id": it["utt_id"],
                                "ref": it["text"],
                                "hyp": "",
                                "cer": 1.0,
                                "confidence": it.get("confidence"),
                                "source": it.get("source"),
                                "model": "firered-asr2-aed",
                                "error": f"batch_fail: {str(exc)[:280]}",
                            }
                            shard_f.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                            in_shard += 1

                for p in tmp_files:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

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
                        f"[firered] done={done_n} (+prev {len(done)}) ok={ok} fail={fail} "
                        f"gCER={gcer:.4f} {rate:.2f} utt/s elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )
                    shard_f.flush()

                cur, fut = nxt, nxt_fut
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

        done_n = ok + fail
        elapsed = time.perf_counter() - t0
        rate = done_n / max(elapsed, 1e-6)
        gcer = total_err / total_ref if total_ref else 0.0
        print(
            f"[firered] done={done_n} (+prev {len(done)}) ok={ok} fail={fail} "
            f"gCER={gcer:.4f} {rate:.2f} utt/s elapsed={elapsed/60:.1f}m",
            flush=True,
        )
    finally:
        shard_f.flush()
        shard_f.close()

    gcer = total_err / total_ref if total_ref else 0.0
    summary = {
        "model": "firered-asr2-aed",
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
