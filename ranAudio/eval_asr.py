#!/usr/bin/env python3
"""CER on AISHELL / WenetSpeech test_net / compact jsonl / car 22."""

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

from ranAudio.car_cases import car_wav_cases
from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, SENSEVOICE, TMP_DIR
from ranAudio.protocol import ASR_SYSTEM, cer, entity_fidelity
from ranAudio.wenet import default_split_paths, iter_supervisions, load_recordings, load_segment_wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="aishell-test")
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "s15_asr" / "projector.pt"))
    ap.add_argument("--encoder", default=str(RUNS_DIR / "s15_asr" / "encoder.pt"))
    ap.add_argument("--encoder-backend", default="sensevoice")
    ap.add_argument("--sensevoice", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--also-car", action="store_true")
    ap.add_argument("--out-json", default=str(TMP_DIR / "minicpm5_asr_eval.json"))
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
    if args.adapter and Path(args.adapter).is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, args.adapter)
    model.eval()

    results = []
    if args.split == "car":
        segs = []
        for case in car_wav_cases():
            segs.append(("car", case["wav"], 0.0, 0.0, case["asr"], True))
    else:
        sup, rec = default_split_paths(args.split)
        recs = load_recordings(rec)
        segs = []
        for i, seg in enumerate(iter_supervisions(sup, recordings=recs, require_audio=True)):
            segs.append((seg.sid, seg.wav, seg.start, seg.duration, seg.text, False))
            if args.max_samples and i + 1 >= args.max_samples:
                break

    cer_sum = 0.0
    n = 0
    fid_sum = 0.0
    n_fid = 0
    for sid, wav_path, start, dur, ref, whole in segs:
        if whole:
            wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        else:
            from ranAudio.wenet import Segment

            wav = load_segment_wav(Segment(sid, Path(wav_path), start, dur, ref))
        hyp = model.generate_text(wav, tools=None, system=ASR_SYSTEM)
        c = cer(ref, hyp)
        cer_sum += c
        n += 1
        fid = entity_fidelity(ref, hyp)
        if fid is not None:
            fid_sum += fid
            n_fid += 1
        results.append({"id": sid, "ref": ref, "hyp": hyp, "cer": c, "entity_fid": fid})
        print(f"  cer={c:.3f} ref={ref!r} hyp={hyp!r}", flush=True)

    if args.also_car and args.split != "car":
        for case in car_wav_cases():
            wav, _ = librosa.load(str(case["wav"]), sr=16000, mono=True)
            hyp = model.generate_text(wav, tools=None, system=ASR_SYSTEM)
            c = cer(case["asr"], hyp)
            results.append({"id": case["id"], "ref": case["asr"], "hyp": hyp, "cer": c, "split": "car"})
            print(f"  car cer={c:.3f} {case['id']} hyp={hyp!r}", flush=True)

    summary = {
        "split": args.split,
        "n": n,
        "cer": (cer_sum / n) if n else None,
        "entity_fidelity": (fid_sum / n_fid) if n_fid else None,
        "results": results,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log = out.with_suffix(".log")
    lines = [f"split={args.split} n={n} cer={summary['cer']} entity_fid={summary['entity_fidelity']}"]
    for row in results:
        lines.append(
            f"cer={row.get('cer')} id={row.get('id')} ref={row.get('ref')!r} hyp={row.get('hyp')!r}"
        )
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}  cer={summary['cer']} n={n} log={log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
