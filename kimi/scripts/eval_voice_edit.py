#!/usr/bin/env python3
"""Evaluate Kimi voice-edit IME: HTTP e2e and text-oracle (rule) modes."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
KIMI = _SCRIPTS.parent
EVAL = KIMI / "eval_voice_edit"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from voice_edit_lib import (  # noqa: E402
    apply_edit_rules,
    cer,
    is_echo_fail,
    norm_text,
    normalize_display,
)


def load_cases(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_wav(row: dict, wav_root: Path, human_root: Path) -> Path | None:
    if row.get("instruction_wav"):
        p = Path(row["instruction_wav"])
        if not p.is_absolute():
            p = EVAL / p
        return p if p.is_file() else None
    hid = human_root / f"{row['id']}.wav"
    if hid.is_file() and hid.stat().st_size > 0:
        return hid
    eng = row.get("engine") or "edge"
    p = wav_root / eng / f"{row['id']}.wav"
    return p if p.is_file() and p.stat().st_size > 0 else None


def utf8_b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def post_edit(base: str, draft: str, wav: Path, timeout: float = 180.0) -> dict:
    data = wav.read_bytes()
    req = urllib.request.Request(
        base.rstrip("/") + "/api/edit",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Filename": wav.name,
            "X-Draft-B64": utf8_b64(draft),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def score_one(row: dict, pred: str, instruction_hyp: str | None, via: str | None) -> dict:
    gold = row["gold"]
    draft = row["draft"]
    instruction = row["instruction"]
    metric = row.get("metric") or ("exact" if row.get("strict", True) else "cer")
    pred_d = normalize_display(pred)
    gold_d = normalize_display(gold)
    draft_d = normalize_display(draft)
    pred_n = norm_text(pred_d)
    gold_n = norm_text(gold_d)
    exact = pred_n == gold_n
    edit_cer = cer(gold_d, pred_d)
    instr_cer = cer(instruction, instruction_hyp or "") if instruction_hyp is not None else None
    echo = is_echo_fail(pred, instruction)
    over = False
    if row["bucket"] == "noop":
        over = pred_n != norm_text(draft_d)
    elif exact is False and gold_n and draft:
        # crude: changed chars that gold kept from draft
        kept = [c for c in draft if c in gold]
        if kept and any(c not in (pred or "") for c in kept if c.strip()):
            # only flag if pred drifted far
            over = edit_cer > 0.5 and not exact
    ok = exact if metric == "exact" else edit_cer <= 0.15
    return {
        "exact": exact,
        "ok": ok,
        "edit_cer": edit_cer,
        "instr_cer": instr_cer,
        "echo_fail": echo,
        "over_edit": over,
        "via": via or "",
        "metric": metric,
    }


def summarize(results: list[dict]) -> dict:
    def agg(items: list[dict]) -> dict:
        n = len(items)
        if not n:
            return {"n": 0}
        out = {
            "n": n,
            "exact": sum(1 for r in items if r["exact"]) / n,
            "ok": sum(1 for r in items if r["ok"]) / n,
            "edit_cer": sum(r["edit_cer"] for r in items) / n,
            "echo_fail": sum(1 for r in items if r["echo_fail"]) / n,
            "over_edit": sum(1 for r in items if r["over_edit"]) / n,
            "tts_skip": sum(1 for r in items if r.get("tts_skip")) / n,
            "error": sum(1 for r in items if r.get("error")) / n,
        }
        ics = [r["instr_cer"] for r in items if r.get("instr_cer") is not None]
        if ics:
            out["instr_cer"] = sum(ics) / len(ics)
        via: dict[str, int] = defaultdict(int)
        for r in items:
            if r.get("via"):
                via[str(r["via"]).split("+")[0]] += 1
        if via:
            out["via"] = {k: v / n for k, v in via.items()}
        return out

    by_bucket: dict[str, list] = defaultdict(list)
    by_engine: dict[str, list] = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r)
        by_engine[r.get("engine") or "unknown"].append(r)
    return {
        "all": agg(results),
        "by_bucket": {k: agg(v) for k, v in sorted(by_bucket.items())},
        "by_engine": {k: agg(v) for k, v in sorted(by_engine.items())},
    }


def gates(summary: dict) -> dict[str, bool]:
    all_ = summary.get("all") or {}
    buckets = summary.get("by_bucket") or {}
    rep = buckets.get("replace") or {}
    tog = buckets.get("toggle") or {}
    nop = buckets.get("noop") or {}
    # combine replace+toggle exact
    rt_n = (rep.get("n") or 0) + (tog.get("n") or 0)
    rt_exact = 0.0
    if rt_n:
        rt_exact = (
            (rep.get("exact", 0) * (rep.get("n") or 0) + tog.get("exact", 0) * (tog.get("n") or 0))
            / rt_n
        )
    return {
        "replace_toggle_exact_ge_0.85": rt_exact >= 0.85 if rt_n else False,
        "noop_exact_ge_0.90": (nop.get("exact") or 0) >= 0.90 if nop.get("n") else False,
        "echo_fail_le_0.05": (all_.get("echo_fail") or 0) <= 0.05,
        "instr_cer_le_0.15": (all_.get("instr_cer") or 1.0) <= 0.15
        if "instr_cer" in all_
        else False,
    }


def render_md(summary: dict, gate: dict, meta: dict) -> str:
    lines = [
        "# Voice-edit eval report",
        "",
        f"- time: {meta.get('time')}",
        f"- mode: {meta.get('mode')}",
        f"- base: {meta.get('base')}",
        f"- n: {summary.get('all', {}).get('n')}",
        "",
        "## Gates",
        "",
    ]
    for k, v in gate.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: `{k}`")
    lines += ["", "## All", "", "```json", json.dumps(summary.get("all"), ensure_ascii=False, indent=2), "```", ""]
    lines += ["## By bucket", ""]
    for k, v in (summary.get("by_bucket") or {}).items():
        lines.append(f"- **{k}**: exact={v.get('exact', 0):.3f} edit_cer={v.get('edit_cer', 0):.3f} n={v.get('n')}")
    lines += ["", "## By engine", ""]
    for k, v in (summary.get("by_engine") or {}).items():
        ic = v.get("instr_cer")
        ic_s = f"{ic:.3f}" if ic is not None else "n/a"
        lines.append(
            f"- **{k}**: exact={v.get('exact', 0):.3f} instr_cer={ic_s} edit_cer={v.get('edit_cer', 0):.3f} n={v.get('n')}"
        )
    lines.append("")
    return "\n".join(lines)


def run_oracle(cases: list[dict]) -> list[dict]:
    out = []
    for row in cases:
        ruled = apply_edit_rules(row["draft"], row["instruction"])
        if ruled is None:
            pred = row["draft"]
            via = "rule_miss"
        else:
            pred, via = ruled
        sc = score_one(row, pred, row["instruction"], via)
        out.append(
            {
                **row,
                "pred": pred,
                "instruction_hyp": row["instruction"],
                "tts_skip": False,
                "error": None,
                "ms": 0,
                **sc,
            }
        )
    return out


def run_http(cases: list[dict], base: str, wav_root: Path, human_root: Path) -> list[dict]:
    out = []
    for i, row in enumerate(cases, 1):
        wav = resolve_wav(row, wav_root, human_root)
        rec = {
            **row,
            "pred": "",
            "instruction_hyp": None,
            "tts_skip": wav is None,
            "error": None,
            "ms": 0,
        }
        if wav is None:
            sc = score_one(row, "", None, None)
            rec.update(sc)
            rec["ok"] = False
            out.append(rec)
            print(f"[{i}/{len(cases)}] SKIP no-wav {row['id']}", flush=True)
            continue
        try:
            t0 = time.time()
            resp = post_edit(base, row["draft"], wav)
            ms = int((time.time() - t0) * 1000)
            pred = (resp.get("text") or "").strip()
            instr_hyp = (resp.get("instruction") or "").strip()
            via = resp.get("edit_via") or ""
            sc = score_one(row, pred, instr_hyp, via)
            rec.update(
                {
                    "pred": pred,
                    "instruction_hyp": instr_hyp,
                    "ms": ms,
                    "error": None,
                    **sc,
                }
            )
            flag = "OK" if sc["ok"] else "BAD"
            print(
                f"[{i}/{len(cases)}] {flag} {row['id']} via={via} "
                f"exact={sc['exact']} cer={sc['edit_cer']:.3f} pred={pred[:40]!r}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            sc = score_one(row, "", None, None)
            rec.update(sc)
            rec["ok"] = False
            print(f"[{i}/{len(cases)}] ERR {row['id']} {rec['error']}", flush=True)
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(EVAL / "cases.jsonl"))
    ap.add_argument("--wav-root", default=str(EVAL / "wavs"))
    ap.add_argument("--human-root", default=str(EVAL / "wavs_human"))
    ap.add_argument("--base", default="http://127.0.0.1:7861")
    ap.add_argument("--mode", choices=("http", "oracle-text"), default="http")
    ap.add_argument("--engines", default="", help="filter engines, comma-separated")
    ap.add_argument("--buckets", default="", help="filter buckets, comma-separated")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default=str(EVAL / "reports" / "latest.json"))
    ap.add_argument("--out-md", default=str(EVAL / "reports" / "latest.md"))
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.engines:
        want = {e.strip() for e in args.engines.split(",") if e.strip()}
        cases = [c for c in cases if (c.get("engine") or "edge") in want]
    if args.buckets:
        want_b = {b.strip() for b in args.buckets.split(",") if b.strip()}
        cases = [c for c in cases if c.get("bucket") in want_b]
    if args.limit:
        cases = cases[: args.limit]

    if args.mode == "oracle-text":
        results = run_oracle(cases)
    else:
        # ready check
        try:
            with urllib.request.urlopen(args.base.rstrip("/") + "/api/ready", timeout=5) as resp:
                ready = json.loads(resp.read().decode("utf-8"))
            if not ready.get("ready"):
                print(f"demo not ready: {ready}", flush=True)
                return 2
        except Exception as exc:  # noqa: BLE001
            print(f"cannot reach demo {args.base}: {exc}", flush=True)
            return 2
        results = run_http(cases, args.base, Path(args.wav_root), Path(args.human_root))

    summary = summarize(results)
    gate = gates(summary)
    meta = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "mode": args.mode,
        "base": args.base,
        "n_cases": len(cases),
    }
    payload = {"meta": meta, "summary": summary, "gates": gate, "results": results}
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_md(summary, gate, meta), encoding="utf-8")
    print(json.dumps({"summary": summary, "gates": gate}, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {out_md}", flush=True)
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
