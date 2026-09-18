#!/usr/bin/env python3
"""Export a compact WenetSpeech jsonl (stream Lhotse; optional S-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import WENET_S_JSONL
from ranAudio.wenet import default_split_paths, export_compact, write_offset_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="wenet-ms", help="wenet-ms | wenet-train | aishell-train | ...")
    ap.add_argument("--subset", default="S", help="S | M | L | all")
    ap.add_argument("--out", default=str(WENET_S_JSONL))
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--index", action="store_true", help="also write .offsets.npy")
    args = ap.parse_args()

    subsets = None if args.subset in ("", "all") else {args.subset}
    sup, rec = default_split_paths(args.split)
    if not sup.is_file() or not rec.is_file():
        print(f"missing manifests {sup} {rec}", flush=True)
        return 1
    out = Path(args.out)
    n = export_compact(
        sup,
        rec,
        out,
        subsets=subsets,
        max_samples=args.max_samples,
        require_audio=True,
    )
    print(f"wrote {n} -> {out} split={args.split} subset={args.subset}", flush=True)
    if args.index:
        idx = out.with_suffix(".offsets.npy")
        k = write_offset_index(out, idx)
        print(f"index {k} -> {idx}", flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
