#!/usr/bin/env python3
"""Offline CS2W spoken→written regression (text only).

Default split: val. test.jsonl is held-out — pass --split test explicitly.
Do NOT tune rules/prompts against test failures.

Usage:
  python scripts/eval_spoken2written.py
  python scripts/eval_spoken2written.py --split val --limit 200
  python scripts/eval_spoken2written.py --split test   # final only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from voice_edit_lib import cer, normalize_display, strip_fillers  # noqa: E402

DEFAULT_DATA = ROOT / "eval_spoken2written" / "CS2W" / "data"
REPORT_DIR = ROOT / "eval_spoken2written" / "reports"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__import__("os").environ.get("CS2W_DATA", str(DEFAULT_DATA))),
    )
    ap.add_argument(
        "--split",
        choices=("train", "val", "test", "smoke"),
        default="val",
        help="Default val. test is held-out — do not tune on it. smoke=local tiny file.",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if args.split == "test":
        print(
            "WARNING: --split test is held-out. Do NOT tune rules/prompts on these results.",
            flush=True,
        )

    if args.split == "smoke":
        path = ROOT / "eval_spoken2written" / "smoke_val.jsonl"
    else:
        # CS2W uses val.jsonl for dev; some mirrors may name it differently
        name = "val.jsonl" if args.split == "val" else f"{args.split}.jsonl"
        path = args.data_dir / name
        if args.split == "val" and not path.is_file():
            alt = args.data_dir / "dev.jsonl"
            if alt.is_file():
                path = alt
    if not path.is_file():
        print(
            f"missing {path}\n"
            "clone https://github.com/guozishan/CS2W → eval_spoken2written/CS2W\n"
            "or: python scripts/eval_spoken2written.py --split smoke",
            flush=True,
        )
        return 1

    rows = _load_jsonl(path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    n = 0
    cer_sum = 0.0
    cer_base = 0.0
    changed = 0
    for row in rows:
        src = (row.get("content") or "").strip()
        ref = (row.get("ref") or "").strip()
        if not src or not ref:
            continue
        n += 1
        base = normalize_display(src)
        ruled = strip_fillers(base)
        pred = ruled[0] if ruled else base
        if ruled:
            changed += 1
        cer_sum += cer(ref, pred)
        cer_base += cer(ref, base)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "split": args.split,
        "path": str(path),
        "n": n,
        "filler_changed": changed,
        "cer_identity": round(cer_base / n, 4) if n else None,
        "cer_strip_fillers": round(cer_sum / n, 4) if n else None,
        "note": "rules-only (strip_fillers). LLM path not run here. test=held-out.",
    }
    out_json = REPORT_DIR / f"rules_{args.split}.json"
    out_md = REPORT_DIR / f"rules_{args.split}.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                f"# CS2W strip_fillers ({args.split})",
                "",
                f"- n={n} filler_changed={changed}",
                f"- CER identity (normalize_display only): {report['cer_identity']}",
                f"- CER after strip_fillers: {report['cer_strip_fillers']}",
                "",
                "test split is held-out; default eval uses val.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
