#!/usr/bin/env python3
"""Filter Tier-A: Qwen == FireRed == official (after punctuation strip).

Uses on-disk SQLite for the Qwen side so L90 (~13M utts) fits in a ~32GiB cgroup.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from audio_slice import strip_punctuation

_HERE = Path(__file__).resolve().parent


def count_qwen_errors(qwen_dir: Path) -> int:
    """Utts whose latest shard row is still a slice/decode failure."""
    latest_error: dict[str, bool] = {}
    for p in sorted(qwen_dir.glob("shard_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                utt_id = row.get("utt_id")
                if utt_id:
                    latest_error[utt_id] = bool(row.get("error"))
    return sum(1 for failed in latest_error.values() if failed)


def retry_qwen_errors(qwen_dir: Path) -> None:
    """Transcribe utts whose shard rows failed (empty hyp). Then compare."""
    n_err = count_qwen_errors(qwen_dir)
    if n_err == 0:
        print("[filter] qwen has no error rows; skip retry", flush=True)
        return
    if qwen_dir.name == "qwen3_asr_L90pct":
        manifest = _HERE / "sample_manifest_rest90.jsonl"
    elif qwen_dir.name == "qwen3_asr_L10pct":
        manifest = _HERE / "sample_manifest.jsonl"
    else:
        raise SystemExit(f"[filter] qwen error rows={n_err} but no manifest for {qwen_dir}")
    print(
        f"[filter] qwen error rows={n_err}; retry Qwen3-ASR before compare",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(_HERE / "run_qwen3_asr_compare.py"),
        "--manifest",
        str(manifest),
        "--model",
        "/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B",
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--batch_size",
        "48",
        "--prefetch_workers",
        "4",
        "--log_every",
        "100",
        "--out_dir",
        str(qwen_dir),
    ]
    subprocess.check_call(cmd)


def iter_shard_rows(out_dir: Path):
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def build_qwen_db(qwen_dir: Path, db_path: Path) -> int:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute(
        """
        CREATE TABLE qwen (
            utt_id TEXT PRIMARY KEY,
            ref TEXT NOT NULL,
            hyp TEXT NOT NULL,
            nref TEXT NOT NULL,
            nq TEXT NOT NULL,
            confidence REAL,
            source TEXT,
            start REAL,
            duration REAL,
            audio_path TEXT
        )
        """
    )
    batch: list[tuple] = []
    n = 0
    for row in iter_shard_rows(qwen_dir):
        uid = row.get("utt_id")
        if not uid or row.get("error"):
            continue
        ref = row.get("ref") or ""
        hyp = row.get("hyp") or ""
        batch.append(
            (
                uid,
                ref,
                hyp,
                strip_punctuation(ref),
                strip_punctuation(hyp),
                row.get("confidence"),
                row.get("source"),
                row.get("start"),
                float(row.get("duration") or 0.0),
                row.get("audio_path"),
            )
        )
        if len(batch) >= 20000:
            con.executemany(
                "INSERT OR REPLACE INTO qwen VALUES (?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            n += len(batch)
            batch.clear()
            if n % 1_000_000 == 0:
                print(f"  qwen indexed {n}...", flush=True)
    if batch:
        con.executemany(
            "INSERT OR REPLACE INTO qwen VALUES (?,?,?,?,?,?,?,?,?,?)",
            batch,
        )
        n += len(batch)
    con.commit()
    con.close()
    return n


def conf_bin(conf) -> str:
    if conf is None:
        return "unknown"
    if conf >= 1.0:
        return "1.0"
    if conf >= 0.95:
        return "[0.95,1)"
    return "<0.95"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen_dir",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/qwen3_asr_L10pct"),
    )
    parser.add_argument(
        "--firered_dir",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/firered_asr_L10pct"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/tier_A_agree3.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/tier_A_summary.json"),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional SQLite path for Qwen index (default under out.parent)",
    )
    parser.add_argument(
        "--keep_db",
        action="store_true",
        help="Keep the temporary Qwen SQLite after filtering",
    )
    args = parser.parse_args()

    retry_qwen_errors(args.qwen_dir)

    db_path = args.db or (args.out.parent / f".tier_a_qwen_{args.qwen_dir.name}.sqlite")
    print(f"Indexing Qwen shards -> {db_path}", flush=True)
    n_qwen = build_qwen_db(args.qwen_dir, db_path)
    print(f"  qwen={n_qwen}", flush=True)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    n_firered = 0
    n_common = 0
    n_a = 0
    n_empty = 0
    dur_a = 0.0
    dur_all = 0.0
    by_source = defaultdict(lambda: {"n": 0, "dur": 0.0})
    by_conf = defaultdict(lambda: {"n": 0, "dur": 0.0})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Streaming FireRed shards and writing Tier-A...", flush=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for row in iter_shard_rows(args.firered_dir):
            uid = row.get("utt_id")
            if not uid or row.get("error"):
                continue
            n_firered += 1
            if n_firered % 1_000_000 == 0:
                print(
                    f"  firered scanned {n_firered}; tier_A={n_a}...",
                    flush=True,
                )
            cur.execute(
                "SELECT ref, hyp, nref, nq, confidence, source, start, duration, audio_path "
                "FROM qwen WHERE utt_id=?",
                (uid,),
            )
            q = cur.fetchone()
            if q is None:
                continue
            n_common += 1
            ref = q["ref"] or row.get("ref") or ""
            hq = q["hyp"] or ""
            hf = row.get("hyp") or ""
            nref = q["nref"]
            nq = q["nq"]
            nf = strip_punctuation(hf)
            dur = float(q["duration"] or row.get("duration") or 0.0)
            dur_all += dur
            if not nref:
                n_empty += 1
                continue
            if nref != nq or nref != nf:
                continue
            n_a += 1
            dur_a += dur
            src = q["source"] or row.get("source") or "other"
            by_source[src]["n"] += 1
            by_source[src]["dur"] += dur
            conf = q["confidence"]
            cb = conf_bin(conf)
            by_conf[cb]["n"] += 1
            by_conf[cb]["dur"] += dur
            out_row = {
                "utt_id": uid,
                "text": ref,
                "text_norm": nref,
                "hyp_qwen": hq,
                "hyp_firered": hf,
                "confidence": conf,
                "source": src,
                "start": q["start"],
                "duration": dur,
                "audio_path": q["audio_path"] or row.get("audio_path"),
                "tier": "A",
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    con.close()
    if not args.keep_db and db_path.exists():
        db_path.unlink()

    summary = {
        "n_qwen": n_qwen,
        "n_firered": n_firered,
        "n_common": n_common,
        "n_empty_ref_skipped": n_empty,
        "tier_A_n": n_a,
        "tier_A_ratio_of_common": (n_a / n_common if n_common else 0.0),
        "tier_A_hours": dur_a / 3600.0,
        "common_hours": dur_all / 3600.0,
        "tier_A_hours_ratio": (dur_a / dur_all if dur_all else 0.0),
        "by_source": {
            k: {
                "n": v["n"],
                "hours": v["dur"] / 3600.0,
                "ratio_n": v["n"] / n_a if n_a else 0.0,
            }
            for k, v in sorted(by_source.items())
        },
        "by_confidence": {
            k: {
                "n": v["n"],
                "hours": v["dur"] / 3600.0,
                "ratio_n": v["n"] / n_a if n_a else 0.0,
            }
            for k, v in sorted(by_conf.items())
        },
        "out": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
