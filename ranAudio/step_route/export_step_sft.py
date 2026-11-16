#!/usr/bin/env python3
"""Export MiniCPM S2 jsonl → Step-Audio SFT jsonl (messages + absolute wav).

Source (default): data/s2_audio_boundary.jsonl
Outputs under data/:
  step_s2_route.jsonl   # FC+BASE only
  step_s2_chat.jsonl    # NOFC chat
  step_s2_asr.jsonl     # NOFC asr / 听写
  step_s2_mix.jsonl     # all three
  step_s2_test.jsonl    # from s2_test.jsonl

Each row:
  {
    "id", "phase", "route", "kind", "user", "target", "wav",
    "messages": [
      {"role": "system", "content": ROUTE_SYSTEM},
      {"role": "human", "content": [{"type": "audio", "audio": "<abs wav>"}]},
      {"role": "assistant", "content": "<target>"}
    ]
  }
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

from ranAudio.paths import DATA_DIR, resolve_wav
from ranAudio.step_route.protocol_step import ROUTE_SYSTEM


def _phase(row: dict) -> str:
    route = (row.get("route") or "").strip()
    if route == "fc":
        return "route"
    kind = (row.get("kind") or "").strip()
    if kind == "asr":
        return "asr"
    # nofc without kind: treat chat (boundary table uses kind=chat|asr)
    return "chat"


def _row_to_step(row: dict, *, skip_missing: bool) -> dict | None:
    wav_rel = row.get("wav") or ""
    wav_path = resolve_wav(wav_rel)
    if not wav_path.is_file():
        if skip_missing:
            return None
        raise FileNotFoundError(wav_path)
    target = (row.get("target") or "").strip()
    if not target:
        return None
    phase = _phase(row)
    messages = [
        {"role": "system", "content": ROUTE_SYSTEM},
        {
            "role": "human",
            "content": [{"type": "audio", "audio": str(wav_path.resolve())}],
        },
        {"role": "assistant", "content": target},
    ]
    out = {
        "id": row.get("id"),
        "phase": phase,
        "route": row.get("route"),
        "kind": row.get("kind"),
        "venue": row.get("venue"),
        "user": row.get("user"),
        "target": target,
        "wav": str(wav_path.resolve()),
        "wav_rel": wav_rel,
        "source": row.get("source"),
        "messages": messages,
    }
    if row.get("tool"):
        out["tool"] = row["tool"]
    return out


def convert(
    src: Path,
    out_dir: Path,
    *,
    skip_missing: bool = True,
    prefix: str = "step_s2",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {"route": [], "chat": [], "asr": [], "mix": []}
    missing = 0
    bad_target = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item = _row_to_step(row, skip_missing=skip_missing)
            if item is None:
                if not (row.get("target") or "").strip():
                    bad_target += 1
                else:
                    missing += 1
                continue
            buckets[item["phase"]].append(item)
            buckets["mix"].append(item)

    written = {}
    for name in ("route", "chat", "asr", "mix"):
        path = out_dir / f"{prefix}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in buckets[name]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        written[name] = {"path": str(path), "n": len(buckets[name])}
    return {
        "source": str(src),
        "written": written,
        "missing_wav": missing,
        "bad_target": bad_target,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=DATA_DIR / "s2_audio_boundary.jsonl",
    )
    ap.add_argument(
        "--test-src",
        type=Path,
        default=DATA_DIR / "s2_test.jsonl",
    )
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--prefix", default="step_s2")
    ap.add_argument("--fail-missing", action="store_true")
    args = ap.parse_args()

    summary = convert(
        args.src,
        args.out_dir,
        skip_missing=not args.fail_missing,
        prefix=args.prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.test_src.is_file():
        test_summary = convert(
            args.test_src,
            args.out_dir,
            skip_missing=not args.fail_missing,
            prefix=f"{args.prefix}_test_tmp",
        )
        # Flatten test into single file (mix of tmp).
        tmp_mix = args.out_dir / f"{args.prefix}_test_tmp_mix.jsonl"
        test_out = args.out_dir / f"{args.prefix}_test.jsonl"
        n = 0
        if tmp_mix.is_file():
            lines = tmp_mix.read_text(encoding="utf-8").splitlines()
            test_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n = len(lines)
            for p in args.out_dir.glob(f"{args.prefix}_test_tmp_*.jsonl"):
                p.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "test_source": str(args.test_src),
                    "test_out": str(test_out),
                    "n": n,
                    "missing_wav": test_summary.get("missing_wav"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
