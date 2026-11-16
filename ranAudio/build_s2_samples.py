#!/usr/bin/env python3
"""Assemble data/s2_audio.jsonl from car + toolish + old chat + new TTS + Wenet.

Car *_000 wavs are holdout and not written. Missing wavs are skipped.
Labels are format_fc_base / format_nofc only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.build_judge_samples import default_chat_reply
from ranAudio.car_cases import TEST_TOOLS
from ranAudio.paths import (
    CAR_TEXTS,
    CAR_WAVS,
    CHAT_TEXTS,
    CHAT_WAVS,
    S2_AUDIO_JSONL,
    S2_TTS_JOBS,
    S2_TTS_WAVS,
    S2_WENET_CHAT,
    TOOLISH_TEXTS,
    rel_wav,
)
from ranAudio.protocol import format_fc_base, format_nofc
from ranAudio.s2_catalog import read_jsonl, write_jsonl


def _wav_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _is_car_holdout(sid: str) -> bool:
    if not sid.endswith("_000"):
        return False
    stem = sid[: -len("_000")]
    return stem in set(TEST_TOOLS)


def _from_car() -> list[dict]:
    out: list[dict] = []
    for row in read_jsonl(CAR_TEXTS):
        sid = str(row.get("id") or "")
        if _is_car_holdout(sid):
            continue
        wav = CAR_WAVS / f"{sid}.wav"
        if not _wav_ok(wav):
            continue
        text = str(row.get("text") or "").strip()
        out.append(
            {
                "id": sid,
                "wav": rel_wav(wav),
                "user": text,
                "target": format_fc_base(text),
                "route": "fc",
                "venue": "base",
                "source": "car",
                "engine": "edge-tts",
                "voice": row.get("voice"),
                "tool": row.get("tool"),
            }
        )
    return out


def _from_text_wavs(texts: Path, wav_dir: Path, *, fc: bool, source: str) -> list[dict]:
    out: list[dict] = []
    for row in read_jsonl(texts):
        sid = str(row.get("id") or "")
        wav = wav_dir / f"{sid}.wav"
        if not _wav_ok(wav):
            continue
        text = str(row.get("text") or "").strip()
        rec = {
            "id": sid,
            "wav": rel_wav(wav),
            "user": text,
            "source": source,
            "engine": "edge-tts",
        }
        if fc:
            rec["target"] = format_fc_base(text)
            rec["route"] = "fc"
            rec["venue"] = "base"
        else:
            reply = str(row.get("reply") or default_chat_reply(text))
            rec["target"] = format_nofc(reply)
            rec["route"] = "nofc"
            rec["kind"] = "chat"
            rec["reply"] = reply
        out.append(rec)
    return out


def _from_tts_jobs(jobs_path: Path, wav_dir: Path) -> list[dict]:
    out: list[dict] = []
    for row in read_jsonl(jobs_path):
        sid = str(row.get("id") or "")
        wav = wav_dir / f"{sid}.wav"
        if not _wav_ok(wav):
            continue
        user = str(row.get("user") or "").strip()
        rec = {
            "id": sid,
            "wav": rel_wav(wav),
            "user": user,
            "source": "tts_new",
            "engine": row.get("engine"),
            "voice": row.get("voice"),
        }
        bucket = str(row.get("bucket") or "")
        if bucket == "fc":
            rec["target"] = format_fc_base(user)
            rec["route"] = "fc"
            rec["venue"] = "base"
        elif bucket == "asr":
            body = str(row.get("body") or "").strip()
            rec["target"] = format_nofc(body)
            rec["route"] = "nofc"
            rec["kind"] = "asr"
            rec["body"] = body
            rec["trap"] = bool(row.get("trap"))
        else:
            reply = str(row.get("reply") or default_chat_reply(user))
            rec["target"] = format_nofc(reply)
            rec["route"] = "nofc"
            rec["kind"] = "chat"
            rec["reply"] = reply
        if row.get("instruct"):
            rec["instruct"] = row["instruct"]
        if row.get("ref_wav"):
            rec["ref_wav"] = row["ref_wav"]
        out.append(rec)
    return out


def _from_wenet(path: Path) -> list[dict]:
    out: list[dict] = []
    for row in read_jsonl(path):
        rec = dict(row)
        rec.setdefault("route", "nofc")
        rec.setdefault("kind", "chat")
        rec.setdefault("source", "wenet_s")
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S2_AUDIO_JSONL))
    ap.add_argument("--jobs", default=str(S2_TTS_JOBS))
    ap.add_argument("--wav-dir", default=str(S2_TTS_WAVS))
    ap.add_argument("--wenet", default=str(S2_WENET_CHAT))
    args = ap.parse_args()

    rows: list[dict] = []
    rows.extend(_from_car())
    rows.extend(_from_text_wavs(TOOLISH_TEXTS, CHAT_WAVS, fc=True, source="toolish"))
    rows.extend(_from_text_wavs(CHAT_TEXTS, CHAT_WAVS, fc=False, source="chat_old"))
    rows.extend(_from_tts_jobs(Path(args.jobs), Path(args.wav_dir)))
    rows.extend(_from_wenet(Path(args.wenet)))

    seen: set[str] = set()
    uniq: list[dict] = []
    for rec in rows:
        sid = str(rec.get("id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        uniq.append(rec)

    out = Path(args.out)
    n = write_jsonl(out, uniq)
    buckets = Counter()
    for rec in uniq:
        if rec.get("route") == "fc":
            buckets["fc"] += 1
        elif rec.get("kind") == "asr":
            buckets["asr"] += 1
        else:
            buckets["chat"] += 1
    sources = Counter(str(r.get("source")) for r in uniq)
    engines = Counter(str(r.get("engine")) for r in uniq)
    summary = {
        "n": n,
        "buckets": dict(buckets),
        "sources": dict(sources),
        "engines": dict(engines),
        "out": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (out.with_suffix(".meta.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
