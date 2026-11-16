#!/usr/bin/env python3
"""Screen ~20 Wenet questions, MiniCPM5 T2T replies, write S2 NOFC chat preview."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import DATA_DIR, MINICPM5, MINICPMO, SENSEVOICE, WENET_S_JSONL, rel_wav
from ranAudio.protocol import format_nofc, needs_external, norm_text
from ranAudio.wenet import Segment, load_segment_wav

Q_MARK = ("吗", "呢", "什么", "怎么", "为什么", "是不是", "能不能", "好不好", "你是谁")
DRAMA = (
    "开枪",
    "警官",
    "起义",
    "革命",
    "皇上",
    "共产党",
    "方志敏",
    "铁将军",
    "军阀",
    "八路军",
    "同志",
    "导演",
    "主演",
    "剧组",
    "润之",
    "大哥",
    "崔叔",
    "春明",
    "床头",
    "瞒得了",
    "烦她",
    "被抓",
    "衣服",
    "躲",
)
DIALOGUE_PRON = ("他", "她", "你们", "咱们")
T2T_SYSTEM = (
    "你是语音助手。用一两句中文直接回答用户。"
    "不要复述用户的原话，不要输出工具、JSON 或 XML。"
)
_FN_RE = re.compile(r"<function|</function>|tool_call", re.I)


def _ok_src(text: str, duration: float) -> bool:
    t = (text or "").strip()
    if not (1.6 <= duration <= 5.5):
        return False
    if not (8 <= len(t) <= 22):
        return False
    if needs_external(t):
        return False
    if any(k in t for k in DRAMA):
        return False
    if any(p in t for p in DIALOGUE_PRON) and "是什么意思" not in t:
        return False
    if not any(k in t for k in Q_MARK):
        return False
    return True


def _ok_reply(user: str, reply: str) -> bool:
    r = (reply or "").strip()
    if not (4 <= len(r) <= 80):
        return False
    if _FN_RE.search(r):
        return False
    if "<|" in r or "<backend>" in r:
        return False
    nu, nr = norm_text(user), norm_text(r)
    if not nr or nu == nr:
        return False
    if nu and nu in nr and len(nu) >= 6:
        return False
    return True


def _collect(src: Path, pool: int, scan: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= scan or len(out) >= pool:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if text in seen or not _ok_src(text, float(row.get("duration") or 0)):
                continue
            seen.add(text)
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WENET_S_JSONL))
    ap.add_argument("--out", default=str(DATA_DIR / "s2_wenet_chat_preview.jsonl"))
    ap.add_argument("--wav-dir", default=str(DATA_DIR / "s2_wenet_chat_preview_wavs"))
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument("--pool", type=int, default=80)
    ap.add_argument("--scan", type=int, default=400000)
    args = ap.parse_args()

    cands = _collect(Path(args.src), args.pool, args.scan)
    print(f"candidates {len(cands)} from {args.src}", flush=True)
    if not cands:
        return 1

    model = MiniCPM5Audio.from_pretrained(
        MINICPM5,
        MINICPMO,
        encoder_backend="sensevoice",
        sensevoice_path=SENSEVOICE,
    )
    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    for row in cands:
        if len(kept) >= args.keep:
            break
        text = str(row["text"]).strip()
        try:
            reply = model.generate_from_text(text, tools=None, system=T2T_SYSTEM, max_new_tokens=80)
        except Exception as exc:  # noqa: BLE001
            print(f"skip gen {text!r}: {exc}", flush=True)
            continue
        reply = (reply or "").strip()
        if not _ok_reply(text, reply):
            print(f"drop {text!r} -> {reply!r}", flush=True)
            continue
        sid = f"wenet_chat_{len(kept):02d}"
        seg = Segment(
            sid=str(row.get("id") or sid),
            wav=Path(str(row["wav"])),
            start=float(row["start"]),
            duration=float(row["duration"]),
            text=text,
        )
        wav = load_segment_wav(seg, sr=16000)
        if wav.size < 1600:
            print(f"drop short wav {text!r}", flush=True)
            continue
        wav_path = wav_dir / f"{sid}.wav"
        peak = float(np.max(np.abs(wav))) or 1.0
        sf.write(str(wav_path), (wav / max(peak, 1e-6) * 0.9).astype(np.float32), 16000)
        rec = {
            "id": sid,
            "wav": rel_wav(wav_path),
            "user": text,
            "target": format_nofc(reply),
            "route": "nofc",
            "kind": "chat",
            "source": "wenet_s",
            "wenet_id": row.get("id"),
            "reply": reply,
        }
        kept.append(rec)
        print(f"keep {sid} {text!r} -> {reply!r}", flush=True)

    out = Path(args.out)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(f"wrote {len(kept)} -> {out}", flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
