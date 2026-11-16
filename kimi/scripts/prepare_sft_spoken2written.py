#!/usr/bin/env python3
"""Build text SFT jsonl (messages) from CS2W expanded train/val.

Never touches test. Input: expanded/{train,val}.jsonl with content/ref.
Output: eval_spoken2written/sft/{train,val}.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KIMI = Path(__file__).resolve().parents[1]
EXPANDED = KIMI / "eval_spoken2written" / "expanded"
OUT_DIR = KIMI / "eval_spoken2written" / "sft"

# Keep short & stable — must match sft_spoken2written.py / future demo adapter.
USER_TMPL = "口语转书面：\n{content}"


def row_to_messages(content: str, ref: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": USER_TMPL.format(content=content.strip())},
            {"role": "assistant", "content": ref.strip()},
        ]
    }


def convert_split(src: Path, dst: Path, *, skip_identical: bool) -> dict:
    n_in = n_out = n_skip_empty = n_skip_same = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            content = (row.get("content") or "").strip()
            ref = (row.get("ref") or "").strip()
            if not content or not ref:
                n_skip_empty += 1
                continue
            if skip_identical and content == ref:
                n_skip_same += 1
                continue
            out = row_to_messages(content, ref)
            out["id"] = row.get("id")
            out["variant"] = row.get("variant")
            out["split"] = row.get("split")
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    return {
        "src": str(src),
        "dst": str(dst),
        "n_in": n_in,
        "n_out": n_out,
        "n_skip_empty": n_skip_empty,
        "n_skip_same": n_skip_same,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expanded-dir", type=Path, default=EXPANDED)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument(
        "--keep-identical",
        action="store_true",
        help="Keep content==ref pairs (default: drop; identity is easy without SFT)",
    )
    args = ap.parse_args()

    skip_identical = not args.keep_identical
    stats = {"splits": {}, "user_tmpl": USER_TMPL, "skip_identical": skip_identical}
    for name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = args.expanded_dir / f"{name}.jsonl"
        if not src.is_file():
            print(f"ERROR missing {src}", file=sys.stderr)
            return 1
        dst = args.out_dir / f"{name}.jsonl"
        st = convert_split(src, dst, skip_identical=skip_identical)
        stats["splits"][name] = st
        print(
            f"{name}: {st['n_in']} → {st['n_out']} "
            f"(skip_empty={st['n_skip_empty']} skip_same={st['n_skip_same']}) → {dst}"
        )

    man = args.out_dir / "manifest.json"
    man.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
