#!/usr/bin/env python3
"""Summarize Qwen / FireRed shard results vs official WenetSpeech labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from audio_slice import cer_chars, strip_punctuation


def load_shards(out_dir: Path) -> Dict[str, dict]:
    by_id: Dict[str, dict] = {}
    for p in sorted(out_dir.glob("shard_*.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_id[row["utt_id"]] = row
    return by_id


def aggregate(rows: Iterable[dict]) -> dict:
    total_err = 0
    total_ref = 0
    n = 0
    n_fail = 0
    by_source = defaultdict(lambda: {"err": 0, "ref": 0, "n": 0})
    by_conf = defaultdict(lambda: {"err": 0, "ref": 0, "n": 0})

    for r in rows:
        n += 1
        if r.get("error") and not r.get("hyp"):
            n_fail += 1
        ref = r.get("ref") or ""
        hyp = r.get("hyp") or ""
        # recompute to be consistent
        cer, errs, reflen = cer_chars(ref, hyp)
        total_err += errs
        total_ref += reflen
        src = r.get("source") or "other"
        by_source[src]["err"] += errs
        by_source[src]["ref"] += reflen
        by_source[src]["n"] += 1
        conf = r.get("confidence")
        if conf is None or conf < 0:
            conf_bin = "unknown"
        elif conf >= 1.0:
            conf_bin = "1.0"
        elif conf >= 0.95:
            conf_bin = "[0.95,1)"
        else:
            conf_bin = "<0.95"
        by_conf[conf_bin]["err"] += errs
        by_conf[conf_bin]["ref"] += reflen
        by_conf[conf_bin]["n"] += 1

    def pack(d):
        return {
            k: {
                "n": v["n"],
                "cer": (v["err"] / v["ref"] if v["ref"] else 0.0),
            }
            for k, v in sorted(d.items())
        }

    return {
        "n": n,
        "n_fail": n_fail,
        "global_cer": (total_err / total_ref if total_ref else 0.0),
        "total_ref_chars": total_ref,
        "by_source": pack(by_source),
        "by_confidence": pack(by_conf),
    }


def pairwise(qwen: Dict[str, dict], firered: Dict[str, dict]) -> dict:
    common = set(qwen) & set(firered)
    both_exact = 0
    qwen_only = 0
    firered_only = 0
    neither = 0
    q_vs_f_err = 0
    q_vs_f_ref = 0

    for uid in common:
        ref = strip_punctuation(qwen[uid].get("ref") or "")
        hq = strip_punctuation(qwen[uid].get("hyp") or "")
        hf = strip_punctuation(firered[uid].get("hyp") or "")
        q_ok = hq == ref
        f_ok = hf == ref
        if q_ok and f_ok:
            both_exact += 1
        elif q_ok and not f_ok:
            qwen_only += 1
        elif f_ok and not q_ok:
            firered_only += 1
        else:
            neither += 1
        cer, errs, reflen = cer_chars(hq, hf)  # treat qwen as ref for model-model
        # better: edit distance between hyps normalized by max len
        from audio_slice import edit_distance

        if not hq and not hf:
            pass
        else:
            denom = max(len(hq), len(hf), 1)
            q_vs_f_err += edit_distance(hq, hf)
            q_vs_f_ref += denom

    n = len(common) or 1
    return {
        "n_common": len(common),
        "exact_match_both_vs_ref": both_exact / n,
        "exact_match_qwen_only": qwen_only / n,
        "exact_match_firered_only": firered_only / n,
        "exact_match_neither": neither / n,
        "qwen_vs_firered_cer": (q_vs_f_err / q_vs_f_ref if q_vs_f_ref else 0.0),
    }


def to_md(summary: dict) -> str:
    lines = ["# WenetSpeech L 10% ASR compare summary", ""]
    for name in ("qwen", "firered"):
        if name not in summary:
            continue
        s = summary[name]
        lines.append(f"## {name}")
        lines.append(f"- n={s['n']} fail={s['n_fail']} global_CER={s['global_cer']:.4f}")
        lines.append("- by source:")
        for k, v in s["by_source"].items():
            lines.append(f"  - {k}: n={v['n']} CER={v['cer']:.4f}")
        lines.append("- by confidence:")
        for k, v in s["by_confidence"].items():
            lines.append(f"  - {k}: n={v['n']} CER={v['cer']:.4f}")
        lines.append("")
    if "pairwise" in summary:
        p = summary["pairwise"]
        lines.append("## pairwise (common utts)")
        for k, v in p.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen_dir",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/qwen3_asr_L10pct"),
    )
    parser.add_argument(
        "--firered_dir",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/firered_asr_L10pct"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/cmy/cmy/ranAudio/wenetSpeechL/outputs/compare_summary.md"),
    )
    args = parser.parse_args()

    summary = {}
    qwen = load_shards(args.qwen_dir) if args.qwen_dir.exists() else {}
    firered = load_shards(args.firered_dir) if args.firered_dir.exists() else {}
    if qwen:
        summary["qwen"] = aggregate(qwen.values())
    if firered:
        summary["firered"] = aggregate(firered.values())
    if qwen and firered:
        summary["pairwise"] = pairwise(qwen, firered)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = to_md(summary)
    args.out.write_text(md, encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"wrote {args.out} and {json_path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
