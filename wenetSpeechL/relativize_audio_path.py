#!/usr/bin/env python3
"""Rewrite `audio_path` to a path relative to the WenetSpeech root.

Example:
  /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/podcast/B00001/x.opus
  -> audio/train/podcast/B00001/x.opus

Works in place via `<file>.new` + atomic rename, so an interrupted run never
leaves a half-written shard behind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY"
MARKER = "audio/"


def to_relative(path: str, root: str) -> str:
    if not path:
        return path
    if not path.startswith("/"):
        return path  # already relative
    if path.startswith(root.rstrip("/") + "/"):
        return path[len(root.rstrip("/")) + 1 :]
    i = path.find("/" + MARKER)
    if i >= 0:
        return path[i + 1 :]
    if path.startswith(MARKER):
        return path
    return path  # unknown layout: leave untouched (reported by caller)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    for src in args.files:
        tmp = src.with_suffix(src.suffix + ".new")
        n = 0
        changed = 0
        left_abs = 0
        with src.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ap_old = row.get("audio_path") or ""
                ap_new = to_relative(ap_old, args.root)
                if ap_new != ap_old:
                    row["audio_path"] = ap_new
                    changed += 1
                if ap_new.startswith("/"):
                    left_abs += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        tmp.replace(src)
        print(f"{src.name}: rows={n} changed={changed} still_absolute={left_abs}", flush=True)


if __name__ == "__main__":
    main()
