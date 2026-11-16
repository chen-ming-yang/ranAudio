#!/usr/bin/env python3
"""Synthesize CS2W expanded `content` with multiple TTS engines → 16 kHz mono wavs.

Engines: edge | qwen_custom | qwen_design (via ranAudio.gen_s2_tts).
Never reads test.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

KIMI = Path(__file__).resolve().parents[1]
ROOT = KIMI.parent
EXPANDED = KIMI / "eval_spoken2written" / "expanded"
WAV_ROOT = KIMI / "eval_spoken2written" / "wavs"
FAIL_LOG = KIMI / "eval_spoken2written" / "tts_fail.jsonl"
INDEX_PATH = EXPANDED / "tts_index.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ranAudio._boot  # noqa: F401
from ranAudio import gen_s2_tts as tts  # noqa: E402

EDGE_VOICES = (
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunjianNeural",
)
CUSTOM_SPEAKERS = ("Vivian", "Serena", "Uncle_Fu", "Dylan")
DESIGN_INSTRUCTS = (
    "成年男声，平稳口语，语速中等，吐字清楚",
    "年轻女声，自然随和，像日常对话",
    "中年女声，略沙哑，语气平静",
)


def _voice_key(voice: str) -> str:
    return re_sub_safe(voice)


def re_sub_safe(voice: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in voice)[:48]


def load_expanded(splits: list[str], limit: int) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        if split == "test":
            raise SystemExit("refusing test split")
        path = EXPANDED / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run expand_cs2w.py first")
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def build_jobs(rows: list[dict], engines: list[str]) -> list[dict]:
    """One job per (row, engine, voice/style). id embeds voice key for wav name."""
    jobs: list[dict] = []
    for i, row in enumerate(rows):
        content = (row.get("content") or "").strip()
        if not content:
            continue
        base_id = str(row["id"])
        for eng in engines:
            if eng == "edge":
                voice = EDGE_VOICES[i % len(EDGE_VOICES)]
                vk = _voice_key(voice)
                jobs.append(
                    {
                        "id": f"{base_id}__{vk}",
                        "user": content,
                        "engine": "edge",
                        "voice": voice,
                        "src_row_id": base_id,
                        "split": row.get("split"),
                        "variant": row.get("variant"),
                        "ref": row.get("ref"),
                        "content": content,
                    }
                )
            elif eng == "qwen_custom":
                spk = CUSTOM_SPEAKERS[i % len(CUSTOM_SPEAKERS)]
                jobs.append(
                    {
                        "id": f"{base_id}__{spk}",
                        "user": content,
                        "engine": "qwen_custom",
                        "voice": spk,
                        "src_row_id": base_id,
                        "split": row.get("split"),
                        "variant": row.get("variant"),
                        "ref": row.get("ref"),
                        "content": content,
                    }
                )
            elif eng == "qwen_design":
                instruct = DESIGN_INSTRUCTS[i % len(DESIGN_INSTRUCTS)]
                vk = f"d{i % len(DESIGN_INSTRUCTS)}"
                jobs.append(
                    {
                        "id": f"{base_id}__{vk}",
                        "user": content,
                        "engine": "qwen_design",
                        "voice": vk,
                        "instruct": instruct,
                        "src_row_id": base_id,
                        "split": row.get("split"),
                        "variant": row.get("variant"),
                        "ref": row.get("ref"),
                        "content": content,
                    }
                )
            else:
                print(f"WARN skip unknown engine {eng}", flush=True)
    return jobs


def _append_index(recs: list[dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    # rewrite full index from existing wavs + this run metadata would be heavy;
    # append unique lines for this batch (idempotent enough with resume)
    seen: set[str] = set()
    if INDEX_PATH.is_file():
        with INDEX_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line).get("wav", ""))
                except Exception:  # noqa: BLE001
                    pass
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        for r in recs:
            wav = r.get("wav") or ""
            if wav in seen:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            seen.add(wav)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument(
        "--engines",
        default="edge,qwen_custom,qwen_design",
        help="comma list: edge,qwen_custom,qwen_design",
    )
    ap.add_argument("--limit", type=int, default=0, help="limit expanded text rows (0=all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(__import__("os").environ.get("CS2W_TTS_BATCH", "16")),
        help="GPU micro-batch for qwen_custom / qwen_design (raise util on H800)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wav-root", type=Path, default=WAV_ROOT)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if "test" in splits:
        print("ERROR: refusing test", flush=True)
        return 2

    rows = load_expanded(splits, args.limit)
    jobs = build_jobs(rows, engines)
    print(f"expanded_rows={len(rows)} jobs={len(jobs)} engines={engines}", flush=True)

    by_eng: dict[str, list[dict]] = defaultdict(list)
    for j in jobs:
        by_eng[j["engine"]].append(j)

    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    index_recs: list[dict] = []

    for eng, eng_jobs in by_eng.items():
        out_dir = args.wav_root / eng
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== engine={eng} n={len(eng_jobs)} -> {out_dir}", flush=True)
        try:
            if eng == "edge":
                ok, skip = asyncio.run(tts.run_edge(eng_jobs, out_dir, args.concurrency))
            elif eng == "qwen_custom":
                ok, skip = tts.run_qwen_custom(
                    eng_jobs, out_dir, args.device, batch_size=args.batch_size
                )
            elif eng == "qwen_design":
                ok, skip = tts.run_qwen_design(
                    eng_jobs, out_dir, args.device, batch_size=args.batch_size
                )
            else:
                print(f"WARN unsupported {eng}", flush=True)
                continue
            print(f"done {eng} ok={ok} skip={skip}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN engine {eng} aborted: {exc}", flush=True)
            with FAIL_LOG.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"engine": eng, "error": str(exc)}, ensure_ascii=False) + "\n"
                )
            continue

        for j in eng_jobs:
            wav = out_dir / f"{j['id']}.wav"
            if wav.is_file() and wav.stat().st_size > 0:
                index_recs.append(
                    {
                        "id": j["id"],
                        "src_row_id": j.get("src_row_id"),
                        "split": j.get("split"),
                        "variant": j.get("variant"),
                        "engine": eng,
                        "voice": j.get("voice"),
                        "instruct": j.get("instruct"),
                        "wav": str(wav.relative_to(KIMI)),
                        "content": j.get("content"),
                        "ref": j.get("ref"),
                    }
                )

    _append_index(index_recs)
    print(f"index += {len(index_recs)} -> {INDEX_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
