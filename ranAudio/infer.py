#!/usr/bin/env python3
"""Offline infer: wav -> judge JSON; Slave only when need_fc=true."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from finetune_fc.slave_client import DEFAULT_SLAVE_URL
from ranAudio.downstream import route
from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, resolve_wav
from ranAudio.tools import judge_system_prompt, parse_judge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "judge" / "projector_final.pt"))
    ap.add_argument("--adapter", default=str(RUNS_DIR / "judge" / "adapter_final"))
    ap.add_argument("--slave-url", default=DEFAULT_SLAVE_URL)
    ap.add_argument(
        "--no-downstream",
        action="store_true",
        help="only run 2B; skip Slave even if need_fc=true",
    )
    args = ap.parse_args()

    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=args.projector if Path(args.projector).is_file() else None,
        encoder_backend="whisper",
    )
    if Path(args.adapter).is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, args.adapter)
    model.eval()

    wav, _ = librosa.load(str(resolve_wav(args.wav)), sr=16000, mono=True)
    text = model.generate_text(wav, tools=None, system=judge_system_prompt())
    parsed = parse_judge(text)
    out: dict = {"raw": text, "judge": parsed}
    if not parsed:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    need_fc = bool(parsed["need_fc"])
    reply = str(parsed.get("reply") or "")
    # Local reply is authoritative when need_fc=false.
    out["reply"] = reply if not need_fc else None
    if not args.no_downstream:
        if need_fc:
            down = route(
                asr=parsed["asr"],
                need_fc=True,
                reply="",
                slave_url=args.slave_url,
            )
            out["downstream"] = down.as_dict()
            out["reply"] = down.reply
        else:
            out["downstream"] = {
                "need_fc": False,
                "asr": parsed["asr"],
                "mode": "chat",
                "reply": reply,
                "ok": True,
                "name": "",
                "arguments": {},
                "result": None,
                "error": None,
                "slave_ms": None,
            }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
