#!/usr/bin/env python3
"""Realtime FastLM benchmark harness.

Modes:
  baseline        historical GPU measurements + optional sync re-run (no new modules)
  offline-events  no-GPU streaming orchestration with recorded model outputs
  step-stream     Step-Audio-2-mini streaming decode backend
  minicpm-probe   MiniCPM5 S2 first-token gate + prefix-chunk probe

GPU modes are imported lazily so this harness runs without any model loaded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

HISTORICAL_BASELINE = Path("/tmp/step_route_results.json")

CAR_WAVS = Path("/cmy/cmy/MiniCPM-o-Demo/finetune_fc/demo_car_fc/wavs")
COMPLEX_WAVS = Path("/tmp/step_complex_fc/wavs")

BASELINE_WAVS: tuple[str, ...] = (
    str(CAR_WAVS / "enable_bluetooth_000.wav"),
    str(COMPLEX_WAVS / "web_search_stock.wav"),
    str(COMPLEX_WAVS / "weather_beijing.wav"),
    str(COMPLEX_WAVS / "datetime_now.wav"),
)

SEGMENTS = (
    "fast_s",
    "slow_s",
    "reinject_s",
    "total_s",
    "fast_loop_s",
    "route_first_ms",
    "emit_ms",
    "first_speak_ms",
)
MODES = ("baseline", "offline-events", "step-stream", "minicpm-probe")


def historical_rows(path: Path = HISTORICAL_BASELINE) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if isinstance(r, dict) and "latency" in r]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for seg in SEGMENTS:
        vals = [float(r["latency"][seg]) for r in rows if seg in r.get("latency", {})]
        if not vals:
            continue
        s = sorted(vals)
        out[seg] = {
            "min": round(s[0], 3),
            "p50": round(statistics.median(vals), 3),
            "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
            "max": round(s[-1], 3),
            "mean": round(statistics.fmean(vals), 3),
        }
    routes: dict[str, int] = {}
    for r in rows:
        key = str(r.get("route") or "unknown")
        routes[key] = routes.get(key, 0) + 1
    out["routes"] = routes
    return out


def collect_sync_baseline(wavs: list[str], *, with_slow: bool = False) -> list[dict[str, Any]]:
    """Re-run the untouched synchronous pipeline (requires Step venv + GPU)."""
    from ranAudio.step_route.fastlm import StepFastLM
    from ranAudio.step_route.pipeline import StepRoutePipeline

    fast = StepFastLM()
    fast.load()
    slow = None
    if with_slow:
        from ranAudio.step_route.slowlm import SlowLMClient

        slow = SlowLMClient()
        slow.start(timeout=300.0)
    pipe = StepRoutePipeline(fast, slow, reinject=with_slow)
    rows: list[dict[str, Any]] = []
    try:
        for wav in wavs:
            t0 = time.perf_counter()
            result = pipe.run(wav)
            d = result.to_dict()
            d["latency"]["total_s"] = round(time.perf_counter() - t0, 3)
            d["mode"] = "baseline-sync-rerun"
            rows.append(d)
            print(f"[bench] {Path(wav).name}: {d['latency']}", flush=True)
    finally:
        if slow is not None:
            slow.close()
    return rows


def run_gpu_mode(mode: str, wavs: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Dispatch to the streaming coordinator with a selected backend."""
    try:
        from ranAudio.step_route.pipeline import RealtimePipeline
    except ImportError as exc:
        raise RuntimeError(
            f"mode={mode} needs the streaming runtime (coordinator / pipeline): {exc}"
        ) from exc
    try:
        pipe = RealtimePipeline(mode=mode, wavs=wavs, out=Path(args.out))
        return pipe.run_all()
    except Exception as exc:
        # Keep the harness runnable when a model env is broken (e.g. Step venv
        # hub/transformers conflict). Record an explicit error row, no fake data.
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[bench] mode={mode} unavailable: {msg}", flush=True)
        return [
            {
                "request_id": "",
                "wav": "",
                "route": "error",
                "mode": mode,
                "latency": {},
                "notes": [f"environment blocked: {msg}"],
            }
        ]


def write_report(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    summary = summarize(rows)
    fallbacks = sum(1 for r in rows if (r.get("fallback") or {}).get("fallback"))
    errors = sum(1 for r in rows if r.get("route") == "error")
    lines = [
        f"# {title}",
        "",
        f"- samples: {summary['n']}",
        f"- routes: {json.dumps(summary.get('routes', {}), ensure_ascii=False)}",
        f"- fidelity fallbacks: {fallbacks}; environment errors: {errors}",
        "",
        "| segment | unit | min | p50 | p95 | max | mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for seg in SEGMENTS:
        s = summary.get(seg)
        if not s:
            continue
        unit = "ms" if seg.endswith("_ms") else "s"
        lines.append(
            f"| {seg} | {unit} | {s['min']} | {s['p50']} | {s['p95']} | {s['max']} | {s['mean']} |"
        )
    base = historical_rows()
    if base and rows:
        b = summarize(base).get("total_s", {})
        cur = summary.get("total_s", {})
        lines += [
            "",
            "## vs baseline (2026-11-02 synchronous Step pipeline)",
            "",
            "| metric | baseline mean | this run mean |",
            "|---|---|---|",
            f"| total_s | {b.get('mean', 'n/a')} | {cur.get('mean', 'n/a')} |",
            f"| fast_s | {summarize(base).get('fast_s', {}).get('mean', 'n/a')} "
            f"| {summary.get('fast_s', {}).get('mean', 'n/a')} |",
        ]
    lines += ["", "## per-sample", "", "| id | wav | route | user_text | fallback | latency |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        fb = (r.get("fallback") or {}).get("fallback")
        lines.append(
            f"| {str(r.get('request_id', ''))[:12]} | {Path(str(r.get('wav', ''))).name} | "
            f"{r.get('route', '')} | {str(r.get('user_text') or '')[:40]} | {fb} | "
            f"{json.dumps(r.get('latency', {}), ensure_ascii=False)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[bench] wrote report {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, default="baseline")
    ap.add_argument("--wav", action="append", default=[], help="repeatable; defaults to BASELINE_WAVS")
    ap.add_argument("--out", default="/tmp/realtime_bench.json")
    ap.add_argument("--report", default="", help="optional markdown report path")
    ap.add_argument("--sync-rerun", action="store_true", help="baseline: re-run synchronous pipeline")
    ap.add_argument("--with-slow", action="store_true", help="baseline re-run: attach SlowLM")
    args = ap.parse_args()

    wavs = [str(Path(w).resolve()) for w in args.wav] or list(BASELINE_WAVS)
    wavs = [w for w in wavs if Path(w).is_file()]
    if not wavs:
        print("no wav files found", file=sys.stderr)
        return 2

    if args.mode == "baseline":
        rows = historical_rows()
        if args.sync_rerun:
            rows += collect_sync_baseline(wavs, with_slow=args.with_slow)
        if not rows:
            print("no historical baseline found; use --sync-rerun", file=sys.stderr)
            return 2
    else:
        rows = run_gpu_mode(args.mode, wavs, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"mode": args.mode, "summary": summarize(rows), "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[bench] mode={args.mode} n={len(rows)} wrote {out}", flush=True)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2), flush=True)
    if args.report:
        write_report(rows, Path(args.report), title=f"FastLM realtime bench — {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
