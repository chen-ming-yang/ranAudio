#!/usr/bin/env python3
"""Split Tier-A agree3 pools (L10 + rest90) into N balanced shards.

Assignment is greedy: each row goes to the shard with the least accumulated
duration, which balances both hours and source (podcast/youtube) mix, and
mixes the two input pools across shards.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_SRC = [
    _HERE / "outputs/tier_A_agree3.jsonl",
    _HERE / "outputs/tier_A_agree3_rest90.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, nargs="+", default=_DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=_HERE / "tier_A_10shards")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--prefix", default="part")
    parser.add_argument(
        "--total-hours",
        type=float,
        default=None,
        help="Skip the pre-scan and use this total (hours) to size shards.",
    )
    args = parser.parse_args()

    n = args.n
    if args.total_hours:
        total_dur = args.total_hours * 3600.0
        print(f"[split] using given total hours={args.total_hours:.3f}", flush=True)
    else:
        total_dur = 0.0
        total_rows = 0
        for src in args.src:
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    total_dur += float(row.get("duration") or 0.0)
                    total_rows += 1
        print(
            f"[split] scanned rows={total_rows} hours={total_dur / 3600:.3f}",
            flush=True,
        )
    print(f"[split] target/shard={total_dur / n / 3600:.3f}h", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = [f"{args.prefix}_{i:02d}.jsonl" for i in range(n)]
    handles = [
        (args.out_dir / name).with_suffix(".jsonl.new").open("w", encoding="utf-8")
        for name in files
    ]
    stats = [
        {"file": files[i], "n": 0, "dur": 0.0, "by_source": {}, "from": {}}
        for i in range(n)
    ]
    heap = [(0.0, i) for i in range(n)]
    heapq.heapify(heap)

    written = 0
    bad = 0
    try:
        for src in args.src:
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        bad += 1
                        continue
                    dur = float(row.get("duration") or 0.0)
                    cur, idx = heapq.heappop(heap)
                    handles[idx].write(json.dumps(row, ensure_ascii=False) + "\n")
                    st = stats[idx]
                    st["n"] += 1
                    st["dur"] += dur
                    src_name = row.get("source") or "other"
                    st["by_source"][src_name] = st["by_source"].get(src_name, 0) + 1
                    st["from"][src.name] = st["from"].get(src.name, 0) + 1
                    heapq.heappush(heap, (cur + dur, idx))
                    written += 1
                    if written % 1_000_000 == 0:
                        print(f"  written {written}...", flush=True)
    finally:
        for h in handles:
            h.close()

    # Rename atomically so a stale part_XX.jsonl is never left behind.
    for i, name in enumerate(files):
        tmp = (args.out_dir / name).with_suffix(".jsonl.new")
        tmp.replace(args.out_dir / name)

    manifest = {
        "n_shards": n,
        "total_rows": written,
        "total_hours": round(total_dur / 3600, 4),
        "hours_per_shard_target": round(total_dur / n / 3600, 4),
        "split_rule": "greedy: each row goes to the shard with least accumulated duration; balances hours + podcast/youtube mix, mixes L10 and rest90",
        "sources_order": [str(p) for p in args.src],
        "bad_lines_skipped": bad,
        "shards": [
            {
                "file": st["file"],
                "n": st["n"],
                "hours": round(st["dur"] / 3600, 4),
                "by_source": dict(sorted(st["by_source"].items())),
                "from": dict(sorted(st["from"].items())),
            }
            for st in stats
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
