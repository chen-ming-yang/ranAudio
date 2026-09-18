#!/usr/bin/env python3
"""S2 route eval: recall/precision, route CER, speak parrot, entity fidelity."""

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

from ranAudio.car_cases import car_wav_cases, chat_wav_cases
from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, SENSEVOICE, TMP_DIR
from ranAudio.protocol import (
    ROUTE_SYSTEM,
    cer,
    entity_fidelity,
    is_parrot,
    parse_stream,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "s2_route" / "projector_final.pt"))
    ap.add_argument("--encoder", default=str(RUNS_DIR / "s15_asr" / "encoder.pt"))
    ap.add_argument("--encoder-backend", default="sensevoice")
    ap.add_argument("--sensevoice", default="")
    ap.add_argument("--adapter", default=str(RUNS_DIR / "s2_route" / "adapter_final"))
    ap.add_argument("--n-chat", type=int, default=22)
    ap.add_argument("--out-json", default=str(TMP_DIR / "minicpm5_route_eval.json"))
    args = ap.parse_args()

    proj = Path(args.projector) if Path(args.projector).is_file() else None
    enc = Path(args.encoder) if Path(args.encoder).is_file() else None
    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=proj,
        encoder_path=enc,
        encoder_backend=args.encoder_backend,
        sensevoice_path=args.sensevoice or SENSEVOICE,
    )
    if Path(args.adapter).is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, args.adapter)
        print(f"loaded adapter {args.adapter}", flush=True)
    model.eval()

    cases = car_wav_cases() + chat_wav_cases(args.n_chat)
    tp = fp = fn = tn = 0
    cer_sum = 0.0
    n_cer = 0
    n_parrot = 0
    n_speak = 0
    n_empty_reply = 0
    fid_sum = 0.0
    n_fid = 0
    results = []
    for case in cases:
        wav, _ = librosa.load(str(case["wav"]), sr=16000, mono=True)
        text = model.generate_text(wav, tools=None, system=ROUTE_SYSTEM)
        parsed = parse_stream(text)
        expect_route = bool(case["need_fc"])
        got_route = parsed["mode"] == "route"
        if expect_route and got_route:
            tp += 1
        elif expect_route and not got_route:
            fn += 1
        elif (not expect_route) and got_route:
            fp += 1
        else:
            tn += 1
        rec = {
            "id": case["id"],
            "expect_route": expect_route,
            "got_mode": parsed["mode"],
            "got_asr": parsed["asr"],
            "got_reply": parsed["reply"],
            "got_filler": parsed["filler"],
            "text": text,
        }
        if expect_route:
            rec["cer"] = cer(case["asr"], parsed["asr"])
            cer_sum += rec["cer"]
            n_cer += 1
            fid = entity_fidelity(case["asr"], parsed["asr"])
            rec["entity_fid"] = fid
            if fid is not None:
                fid_sum += fid
                n_fid += 1
        else:
            rec["parrot"] = is_parrot(parsed["reply"], case["asr"])
            rec["empty_reply"] = not bool(parsed["reply"])
            n_speak += 1
            n_parrot += int(rec["parrot"])
            n_empty_reply += int(rec["empty_reply"])
        results.append(rec)
        print(
            f"  {'OK' if got_route == expect_route else 'BAD'} {case['id']} "
            f"mode={parsed['mode']} expect_route={expect_route}",
            flush=True,
        )

    recall = tp / (tp + fn) if (tp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    summary = {
        "n": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "precision": prec,
        "route_cer": (cer_sum / n_cer) if n_cer else None,
        "entity_fidelity": (fid_sum / n_fid) if n_fid else None,
        "speak_parrot_rate": (n_parrot / n_speak) if n_speak else None,
        "speak_empty_reply_rate": (n_empty_reply / n_speak) if n_speak else None,
        "results": results,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"wrote {out} recall={recall} prec={prec} cer={summary['route_cer']} "
        f"parrot={summary['speak_parrot_rate']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
