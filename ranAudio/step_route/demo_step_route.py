#!/usr/bin/env python3
"""Offline demo: Step-Audio FastLM + Qwen3.5-2B SlowLM (S2 FC/NOFC).

Run with Step-Audio venv (loads FastLM). SlowLM is spawned under qwen3-compare.

  /cmy/cmy/Step-Audio/venv/bin/python -m ranAudio.step_route.demo_step_route \\
      --wav /tmp/step_complex_fc/wavs/web_search_stock.wav \\
      --wav /cmy/cmy/MiniCPM-o-Demo/finetune_fc/demo_car_fc/wavs/enable_bluetooth_000.wav \\
      --out /tmp/step_route_results.json

See ranAudio/STEP_ROUTE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

from ranAudio.step_route.fastlm import StepFastLM
from ranAudio.step_route.pipeline import StepRoutePipeline
from ranAudio.step_route.slowlm import SlowLMClient

CAR_WAVS = Path("/cmy/cmy/MiniCPM-o-Demo/finetune_fc/demo_car_fc/wavs")
COMPLEX_WAVS = Path("/tmp/step_complex_fc/wavs")

DEFAULT_CASES = [
    # expected: nofc-ish chat (may still FC if prompt weak — logged either way)
    ("chat_who", CAR_WAVS / "enable_bluetooth_000.wav"),  # placeholder swapped below
]


def _default_wavs() -> list[Path]:
    cands = [
        CAR_WAVS / "enable_bluetooth_000.wav",
        COMPLEX_WAVS / "web_search_stock.wav",
        COMPLEX_WAVS / "weather_beijing.wav",
        COMPLEX_WAVS / "datetime_now.wav",
    ]
    # Prefer a chat-like wav if present in s2 test set.
    chat = Path("/cmy/cmy/ranAudio/data/s2_test_wavs")
    if chat.is_dir():
        for p in sorted(chat.glob("*.wav"))[:1]:
            cands.insert(0, p)
    return [p for p in cands if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", default=[], help="wav path (repeatable)")
    ap.add_argument("--out", default="/tmp/step_route_results.json")
    ap.add_argument("--no-slow", action="store_true", help="FastLM only")
    ap.add_argument("--no-reinject", action="store_true")
    ap.add_argument("--step-model", default="/cmy/cmy/Step-Audio/Step-Audio-2-mini")
    ap.add_argument("--qwen-model", default="/cmy/cmy/Qwen/Qwen3.5-2B")
    ap.add_argument("--max-cases", type=int, default=4)
    ap.add_argument(
        "--mode",
        choices=("baseline", "offline-events", "step-stream", "minicpm-probe"),
        default="baseline",
        help="baseline keeps the original synchronous pipeline; others use the streaming runtime",
    )
    ap.add_argument("--report", default="", help="optional markdown report for streaming modes")
    args = ap.parse_args()

    wavs = [Path(w) for w in args.wav] if args.wav else _default_wavs()
    wavs = wavs[: max(1, args.max_cases)]
    if not wavs:
        print("no wav files found", file=sys.stderr)
        return 2

    if args.mode != "baseline":
        from ranAudio.step_route.pipeline import RealtimePipeline

        pipe = RealtimePipeline(mode=args.mode, wavs=[str(w) for w in wavs], out=Path(args.out))
        rows = pipe.run_all()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.report:
            from ranAudio.step_route.bench_realtime import write_report

            write_report(rows, Path(args.report), title=f"FastLM realtime demo — {args.mode}")
        print(f"\n[demo] wrote {out} ({len(rows)} rows, mode={args.mode})", flush=True)
        return 0

    fast = StepFastLM(args.step_model)
    fast.load()

    slow = None
    if not args.no_slow:
        slow = SlowLMClient(model=Path(args.qwen_model))
        print("[demo] starting SlowLM worker…", flush=True)
        slow.start(timeout=300.0)
        print("[demo] SlowLM ready", flush=True)

    pipe = StepRoutePipeline(
        fast,
        slow,
        reinject=not args.no_reinject,
        reinject_with_wav=True,
    )

    rows = []
    try:
        for wav in wavs:
            print(f"\n=== {wav} ===", flush=True)
            r = pipe.run(wav)
            d = r.to_dict()
            rows.append(d)
            print(json.dumps(d, ensure_ascii=False, indent=2), flush=True)
    finally:
        if slow is not None:
            slow.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[demo] wrote {out} ({len(rows)} turns)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
