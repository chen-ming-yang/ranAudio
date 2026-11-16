#!/usr/bin/env python3
"""Expand CS2W train/val spoken texts (content) with rule-based variants.

Never touches test.jsonl. Writes kimi/eval_spoken2written/expanded/{train,val}.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

KIMI = Path(__file__).resolve().parents[1]
OUT_DIR = KIMI / "eval_spoken2written" / "expanded"
CS2W_DATA = KIMI / "eval_spoken2written" / "CS2W" / "data"
RAW_MIRRORS = (
    "https://cdn.jsdelivr.net/gh/guozishan/CS2W@main/data/{name}",
    "https://raw.githubusercontent.com/guozishan/CS2W/main/data/{name}",
)

_LEAD_FILLER_RE = re.compile(r"^(?:嗯|呃|哎|欸|唔|哦|那个|就是说)[，,、\s]")
_HAN_BIGRAM_RE = re.compile(r"[\u4e00-\u9fff]{2}")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ensure_split(split: str) -> Path:
    CS2W_DATA.mkdir(parents=True, exist_ok=True)
    name = f"{split}.jsonl"
    path = CS2W_DATA / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    last_err: Exception | None = None
    for tmpl in RAW_MIRRORS:
        url = tmpl.format(name=name)
        try:
            print(f"fetch {url}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "cs2w-expand/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            if len(data) < 100:
                raise RuntimeError(f"too small: {len(data)} bytes")
            path.write_bytes(data)
            print(f"wrote {path} ({len(data)} bytes)", flush=True)
            return path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"WARN fetch failed: {exc}", flush=True)
    raise SystemExit(
        f"missing {path}; fetch failed ({last_err}). "
        "Clone https://github.com/guozishan/CS2W and copy data/train.jsonl data/val.jsonl here."
    )


def _aug_lead_filler(content: str, src_id: int) -> str | None:
    raw = (content or "").strip()
    if not raw or _LEAD_FILLER_RE.match(raw):
        return None
    filler = "嗯，" if (src_id % 2 == 0) else "呃，"
    return filler + raw


def _aug_stutter_dup(content: str) -> str | None:
    raw = (content or "").strip()
    if not raw:
        return None
    m = _HAN_BIGRAM_RE.search(raw)
    if not m:
        return None
    bi = m.group(0)
    if raw[m.start() : m.start() + 4] == bi + bi:
        return None
    out = raw[: m.start()] + bi + bi + raw[m.end() :]
    if out == raw:
        return None
    return out


def expand_row(row: dict, split: str) -> list[dict]:
    src_id = row.get("id")
    content = (row.get("content") or "").strip()
    ref = (row.get("ref") or "").strip()
    if src_id is None or not content or not ref:
        return []
    base = {
        "src_id": src_id,
        "split": split,
        "ref": ref,
        "disfluency_num": row.get("disfluency_num", 0),
        "grammatical_error_num": row.get("grammatical_error_num", 0),
        "colloquial_word_num": row.get("colloquial_word_num", 0),
        "asr_error_num": row.get("asr_error_num", 0),
    }
    out: list[dict] = [
        {**base, "id": f"{split}_{src_id}_orig", "variant": "orig", "content": content}
    ]
    lead = _aug_lead_filler(content, int(src_id))
    if lead:
        out.append(
            {
                **base,
                "id": f"{split}_{src_id}_lead_filler",
                "variant": "lead_filler",
                "content": lead,
            }
        )
    stutter = _aug_stutter_dup(content)
    if stutter and stutter != content:
        out.append(
            {
                **base,
                "id": f"{split}_{src_id}_stutter_dup",
                "variant": "stutter_dup",
                "content": stutter,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="train,val", help="comma list; never test")
    ap.add_argument("--limit", type=int, default=0, help="per-split original row limit (0=all)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "test" in splits:
        print("ERROR: test split is held-out; refusing", flush=True)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"splits": {}, "total_out": 0}
    for split in splits:
        src = _ensure_split(split)
        rows = _load_jsonl(src)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        expanded: list[dict] = []
        for row in rows:
            expanded.extend(expand_row(row, split))
        dst = args.out_dir / f"{split}.jsonl"
        with dst.open("w", encoding="utf-8") as f:
            for rec in expanded:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["splits"][split] = {
            "src": str(src),
            "n_src": len(rows),
            "n_out": len(expanded),
            "path": str(dst),
        }
        summary["total_out"] += len(expanded)
        print(f"{split}: {len(rows)} -> {len(expanded)} ({dst})", flush=True)

    man = args.out_dir / "manifest.json"
    man.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
