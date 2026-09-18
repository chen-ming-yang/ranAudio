#!/usr/bin/env python3
"""Turn-based car FC eval: one wav per tool, name+arguments must match."""

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

from finetune_fc.fc_format import extract_slots_from_text, load_tools
from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import CAR_TEXTS, CAR_WAVS, MINICPM5, MINICPMO, RUNS_DIR, TMP_DIR
from ranAudio.tools import parse_tool_calls

TEST_TOOLS = [
    "enable_bluetooth",
    "disable_bluetooth",
    "navigate_to",
    "cancel_navigation",
    "enable_ac",
    "disable_ac",
    "set_temperature",
    "volume_up",
    "volume_down",
    "set_volume",
    "open_window",
    "close_window",
    "answer_call",
    "reject_call",
    "play_music",
    "pause_music",
    "next_track",
    "previous_track",
    "enable_seat_heating",
    "disable_seat_heating",
    "enable_headlights",
    "disable_headlights",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "fc" / "projector_final.pt"))
    ap.add_argument("--adapter", default=str(RUNS_DIR / "fc" / "adapter_final"))
    ap.add_argument("--out-json", default=str(TMP_DIR / "minicpm5_car_fc_eval.json"))
    args = ap.parse_args()

    tools = load_tools()
    texts = {}
    with CAR_TEXTS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts[str(row.get("id") or "")] = row

    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=args.projector if Path(args.projector).is_file() else None,
        encoder_backend="whisper",
    )
    if Path(args.adapter).is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, args.adapter)
        print(f"loaded adapter {args.adapter}", flush=True)
    model.eval()

    results = []
    n_ok = 0
    for name in TEST_TOOLS:
        wav_path = CAR_WAVS / f"{name}_000.wav"
        if not wav_path.is_file():
            print(f"warn missing {wav_path}", flush=True)
            continue
        row = texts.get(wav_path.stem) or {}
        expect_args = extract_slots_from_text(name, str(row.get("text") or ""))
        wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        text = model.generate_text(wav, tools=tools)
        parsed = parse_tool_calls(text)
        got = parsed[0] if parsed else {}
        got_name = got.get("name")
        got_args = got.get("arguments") if isinstance(got.get("arguments"), dict) else {}
        ok = got_name == name and got_args == expect_args
        tag = "PASS" if ok else "FAIL"
        if ok:
            n_ok += 1
        rec = {
            "wav": wav_path.name,
            "expect": name,
            "expect_args": expect_args,
            "ok": ok,
            "parsed_name": got_name,
            "parsed_args": got_args,
            "text": text,
        }
        results.append(rec)
        print(f"  {tag} {name} parsed={got_name!r} args={got_args!r} text={text!r}", flush=True)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_ok": n_ok, "n": len(results), "results": results}, ensure_ascii=False, indent=2))
    print(f"wrote {out}  {n_ok}/{len(results)} PASS", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
