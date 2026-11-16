#!/usr/bin/env python3
"""Concurrent download of missing Kimi-Audio-7B-Instruct files via ModelScope + aria2c."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = "moonshotai/Kimi-Audio-7B-Instruct"
BASE = f"https://www.modelscope.cn/models/{REPO}/resolve/master"


def missing_files(out_dir: Path) -> list[tuple[str, int]]:
    api = HubApi()
    files = api.get_model_files(REPO, recursive=True)
    need: list[tuple[str, int]] = []
    for f in files:
        if f.get("Type") == "tree":
            continue
        path = f.get("Path") or f.get("Name")
        size = int(f.get("Size") or 0)
        if not path:
            continue
        local = out_dir / path
        if local.is_file() and size > 0 and local.stat().st_size == size:
            continue
        if local.is_file() and size > 0 and local.stat().st_size >= int(size * 0.995):
            continue
        need.append((path, size))
    need.sort(key=lambda x: -x[1])
    return need


def write_aria2_input(need: list[tuple[str, int]], out_dir: Path, list_path: Path) -> None:
    lines: list[str] = []
    for rel, _size in need:
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{rel}"
        lines.append(url)
        lines.append(f"  dir={dest.parent}")
        lines.append(f"  out={dest.name}")
        lines.append("")
    list_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default="/cmy/cmy/ranAudio/kimi/models/Kimi-Audio-7B-Instruct",
    )
    ap.add_argument("--jobs", type=int, default=8, help="parallel files")
    ap.add_argument("--connections", type=int, default=16, help="per-file connections")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need = missing_files(out_dir)
    total = sum(s for _, s in need)
    print(f"missing_files={len(need)} bytes={total} (~{total/1e9:.1f}G)", flush=True)
    for rel, size in need:
        print(f"  {size/1e9:8.3f}G  {rel}", flush=True)
    if not need:
        print("nothing to download", flush=True)
        return 0

    list_path = out_dir / ".aria2_missing.txt"
    write_aria2_input(need, out_dir, list_path)
    print(f"aria2 list -> {list_path}", flush=True)
    if args.dry_run:
        return 0

    cmd = [
        "aria2c",
        "-c",
        "-x",
        str(args.connections),
        "-s",
        str(args.connections),
        "-j",
        str(args.jobs),
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=0",
        "--retry-wait=3",
        "--timeout=60",
        "--connect-timeout=20",
        "--summary-interval=30",
        "--console-log-level=notice",
        "-i",
        str(list_path),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
