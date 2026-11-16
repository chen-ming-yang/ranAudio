#!/usr/bin/env python3
"""Score S2 <|FC|><|BASE|> / <|NOFC|> outputs on a jsonl test set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, S2_TEST_JSONL, SENSEVOICE, TMP_DIR, resolve_wav
from ranAudio.protocol import BASE, FC, NOFC, S2_SYSTEM, cer, is_parrot
from ranAudio.s2_catalog import read_jsonl
from transformers import AutoTokenizer

S15_ENC = RUNS_DIR / "s15_asr" / "b_ms_cont" / "encoder_best.pt"
S15_PROJ = RUNS_DIR / "s15_asr" / "b_ms_cont" / "projector_best.pt"
ADAPTER = RUNS_DIR / "s2_fc_nofc" / "mix" / "adapter_final"


def _bucket(row: dict) -> str:
    if row.get("route") == "fc" or row.get("bucket") == "fc":
        return "fc"
    if row.get("kind") == "asr" or row.get("bucket") == "asr":
        return "asr"
    return "chat"


def _score(text: str, row: dict) -> dict:
    raw = (text or "").strip()
    bucket = _bucket(row)
    bad_tok = any(tok in raw for tok in ("<|LOCAL|>", "<function", "<backend>"))
    if bucket == "fc":
        route_ok = raw.startswith(f"{FC}{BASE}") and not bad_tok
        body = raw[len(FC) + len(BASE) :] if raw.startswith(f"{FC}{BASE}") else raw
        return {
            "bucket": bucket,
            "route_ok": route_ok,
            "cer": cer(str(row.get("user") or ""), body.strip()),
            "parrot": False,
        }
    route_ok = raw.startswith(NOFC) and not raw.startswith(FC) and not bad_tok
    body = raw[len(NOFC) :] if raw.startswith(NOFC) else raw
    body = body.strip()
    if bucket == "asr":
        return {
            "bucket": bucket,
            "route_ok": route_ok,
            "cer": cer(str(row.get("body") or ""), body),
            "parrot": False,
            "trap": bool(row.get("trap")),
        }
    return {
        "bucket": bucket,
        "route_ok": route_ok,
        "cer": None,
        "parrot": is_parrot(body, str(row.get("user") or "")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(S2_TEST_JSONL))
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--encoder", default=str(S15_ENC))
    ap.add_argument("--projector", default=str(S15_PROJ))
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--sensevoice", default="")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-json", default=str(TMP_DIR / "s2_test_eval.json"))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.samples))
    if args.max_samples:
        rows = rows[: args.max_samples]
    tokenizer = None
    tok_dir = Path(args.adapter) / "tokenizer"
    if tok_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
        print(f"eval reuse tokenizer from {tok_dir}", flush=True)
    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=Path(args.projector),
        encoder_path=Path(args.encoder),
        encoder_backend="sensevoice",
        sensevoice_path=args.sensevoice or SENSEVOICE,
        tokenizer=tokenizer,
    )
    adapter = Path(args.adapter)
    if adapter.is_dir():
        llm_ckpt = adapter / "llm_final.safetensors"
        peft_cfg = adapter / "adapter_config.json"
        if llm_ckpt.is_file():
            from safetensors.torch import load_file

            sd = load_file(str(llm_ckpt), device=str(model.device))
            model.llm.load_state_dict(sd)
            print(f"loaded full llm {llm_ckpt}", flush=True)
        elif peft_cfg.is_file() or (adapter / "adapter_model.safetensors").is_file():
            from peft import PeftModel

            model.llm = PeftModel.from_pretrained(model.llm, str(adapter))
            print(f"loaded adapter {adapter}", flush=True)
        else:
            print(f"warn: no llm/adapter weights under {adapter}", flush=True)
    model.eval()

    results = []
    by = defaultdict(list)
    for i, row in enumerate(rows, 1):
        wav_path = resolve_wav(row["wav"])
        wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        text = model.generate_text(wav, tools=None, system=S2_SYSTEM, max_new_tokens=128)
        scored = _score(text, row)
        rec = {
            "id": row.get("id"),
            "source": row.get("source"),
            "user": row.get("user"),
            "target": row.get("target"),
            "hyp": text,
            **scored,
        }
        results.append(rec)
        by[scored["bucket"]].append(rec)
        flag = "OK" if scored["route_ok"] else "BAD"
        extra = ""
        if scored["cer"] is not None:
            extra = f" cer={scored['cer']:.3f}"
        if scored["bucket"] == "chat":
            extra = f" parrot={scored['parrot']}"
        print(f"[{i}/{len(rows)}] {flag} {row.get('id')} {text[:80]!r}{extra}", flush=True)

    summary = {"n": len(results), "buckets": {}}
    for name, items in by.items():
        n = len(items)
        route_ok = sum(1 for r in items if r["route_ok"])
        block = {"n": n, "route_acc": route_ok / n if n else None}
        cers = [r["cer"] for r in items if r["cer"] is not None]
        if cers:
            block["cer"] = sum(cers) / len(cers)
        if name == "chat":
            block["parrot"] = sum(1 for r in items if r["parrot"]) / n if n else None
        if name == "asr":
            traps = [r for r in items if r.get("trap")]
            if traps:
                block["trap_n"] = len(traps)
                block["trap_route_acc"] = sum(1 for r in traps if r["route_ok"]) / len(traps)
        summary["buckets"][name] = block
    summary["route_acc"] = (
        sum(1 for r in results if r["route_ok"]) / len(results) if results else None
    )
    summary["sources"] = dict(Counter(str(r.get("source")) for r in results))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
