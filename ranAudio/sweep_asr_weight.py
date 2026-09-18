#!/usr/bin/env python3
"""§10.5: sweep <asr> loss weight before locking S2. Short runs, then eval_route."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import MINICPM5, MINICPMO, ROUTE_JSONL, RUNS_DIR, TMP_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="0,0.1,0.3,1.0")
    ap.add_argument("--samples", default=str(ROUTE_JSONL))
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "s15_asr" / "projector.pt"))
    ap.add_argument("--encoder", default=str(RUNS_DIR / "s15_asr" / "encoder.pt"))
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out-json", default=str(TMP_DIR / "minicpm5_asr_weight_sweep.json"))
    args = ap.parse_args()

    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    py = sys.executable
    rows = []
    for w in weights:
        tag = str(w).replace(".", "p")
        out_dir = RUNS_DIR / f"s2_sweep_asr{tag}"
        print(f"== train asr_weight={w} -> {out_dir} ==", flush=True)
        train_cmd = [
            py,
            "-m",
            "ranAudio.train_s2_route",
            "--samples",
            args.samples,
            "--llm",
            args.llm,
            "--apm",
            args.apm,
            "--projector",
            args.projector,
            "--encoder",
            args.encoder,
            "--out-dir",
            str(out_dir),
            "--asr-weight",
            str(w),
            "--phase",
            "mix",
            "--epochs",
            str(args.epochs),
            "--max-samples",
            str(args.max_samples),
            "--max-steps",
            str(args.max_steps),
        ]
        subprocess.check_call(train_cmd)
        eval_json = TMP_DIR / f"minicpm5_route_sweep_asr{tag}.json"
        eval_cmd = [
            py,
            "-m",
            "ranAudio.eval_route",
            "--llm",
            args.llm,
            "--apm",
            args.apm,
            "--projector",
            str(out_dir / "projector_final.pt"),
            "--encoder",
            args.encoder,
            "--adapter",
            str(out_dir / "adapter_final"),
            "--out-json",
            str(eval_json),
        ]
        subprocess.check_call(eval_cmd)
        summary = json.loads(eval_json.read_text(encoding="utf-8"))
        row = {
            "asr_weight": w,
            "out_dir": str(out_dir),
            "recall": summary.get("recall"),
            "precision": summary.get("precision"),
            "route_cer": summary.get("route_cer"),
            "speak_parrot_rate": summary.get("speak_parrot_rate"),
            "upload_rate": None,
        }
        n = summary.get("n") or 0
        if n:
            pred_route = summary.get("tp", 0) + summary.get("fp", 0)
            row["upload_rate"] = pred_route / n
        rows.append(row)
        print(f"weight={w} {row}", flush=True)

    # pick knee: prefer recall>=0.95, then lowest parrot, then lowest cer
    def _key(r: dict) -> tuple:
        rec = r.get("recall")
        rec_ok = 0 if (rec is not None and rec >= 0.95) else 1
        parrot = r.get("speak_parrot_rate") if r.get("speak_parrot_rate") is not None else 1.0
        c = r.get("route_cer") if r.get("route_cer") is not None else 1.0
        return (rec_ok, parrot, c)

    pick = min(rows, key=_key) if rows else None
    report = {"runs": rows, "suggested": pick}
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out} suggested={pick}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
