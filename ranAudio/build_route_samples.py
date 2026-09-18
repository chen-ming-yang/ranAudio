#!/usr/bin/env python3
"""Judge jsonl -> §10.1 speak/route tags. Does not rewrite v1.1 judge_audio.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.build_judge_samples import default_chat_reply
from ranAudio.paths import JUDGE_JSONL, ROUTE_JSONL
from ranAudio.protocol import format_route, format_speak

try:
    from finetune_fc.fc_format import ACK_TEMPLATES
except Exception:
    ACK_TEMPLATES = {}


def route_target(row: dict) -> str:
    asr = str(row.get("asr") or row.get("text") or "")
    if row.get("need_fc"):
        tool = str(row.get("tool") or "")
        filler = ACK_TEMPLATES.get(tool, "好的")
        return format_route(asr, filler=filler)
    reply = str(row.get("reply") or default_chat_reply(asr))
    return format_speak(reply)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(JUDGE_JSONL))
    ap.add_argument("--out", default=str(ROUTE_JSONL))
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_file():
        print(f"missing {src}", flush=True)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    n_route = n_speak = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            target = route_target(row)
            rec = dict(row)
            rec["mode"] = "route" if row.get("need_fc") else "speak"
            rec["target"] = target
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["mode"] == "route":
                n_route += 1
            else:
                n_speak += 1
    print(f"wrote route={n_route} speak={n_speak} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
