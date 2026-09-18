#!/usr/bin/env python3
"""Baseline probes: bare-2B text FC, speech FC, ASR/route/reply conflict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from finetune_fc.fc_format import extract_slots_from_text, load_tools
from ranAudio.car_cases import car_wav_cases, chat_wav_cases
from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, TMP_DIR
from ranAudio.protocol import (
    ASR_SYSTEM,
    REPLY_SYSTEM,
    ROUTE_ONLY_SYSTEM,
    ROUTE_SYSTEM,
    cer,
    is_parrot,
    norm_text,
    parse_stream,
)
from ranAudio.tools import fc_system_prompt, parse_tool_calls


def _load_model(
    args,
    *,
    projector: Path | None,
    adapter: Path | None,
    encoder_backend: str = "whisper",
) -> MiniCPM5Audio:
    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=projector if projector and projector.is_file() else None,
        encoder_backend=encoder_backend,
    )
    if adapter and adapter.is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, str(adapter))
        print(f"loaded adapter {adapter}", flush=True)
    model.eval()
    return model


def _eval_text_fc(model: MiniCPM5Audio, cars: list[dict], tools: list) -> dict:
    results = []
    n_ok = 0
    for case in cars:
        expect = case["tool"]
        expect_args = extract_slots_from_text(expect, case["text"])
        text = model.generate_from_text(case["text"], tools=tools, system=fc_system_prompt())
        parsed = parse_tool_calls(text)
        got = parsed[0] if parsed else {}
        got_name = got.get("name")
        got_args = got.get("arguments") if isinstance(got.get("arguments"), dict) else {}
        ok = got_name == expect and got_args == expect_args
        n_ok += int(ok)
        results.append(
            {
                "id": case["id"],
                "expect": expect,
                "expect_args": expect_args,
                "parsed_name": got_name,
                "parsed_args": got_args,
                "ok": ok,
                "text": text,
            }
        )
        print(f"  text_fc {'PASS' if ok else 'FAIL'} {expect} parsed={got_name!r}", flush=True)
    return {"n": len(results), "n_ok": n_ok, "results": results}


def _eval_speech_fc(model: MiniCPM5Audio, cars: list[dict], tools: list) -> dict:
    results = []
    n_ok = 0
    for case in cars:
        expect = case["tool"]
        expect_args = extract_slots_from_text(expect, case["text"])
        wav, _ = librosa.load(str(case["wav"]), sr=16000, mono=True)
        text = model.generate_text(wav, tools=tools, system=fc_system_prompt())
        parsed = parse_tool_calls(text)
        got = parsed[0] if parsed else {}
        got_name = got.get("name")
        got_args = got.get("arguments") if isinstance(got.get("arguments"), dict) else {}
        ok = got_name == expect and got_args == expect_args
        n_ok += int(ok)
        results.append(
            {
                "id": case["id"],
                "expect": expect,
                "expect_args": expect_args,
                "parsed_name": got_name,
                "parsed_args": got_args,
                "ok": ok,
                "text": text,
            }
        )
        print(f"  speech_fc {'PASS' if ok else 'FAIL'} {expect} parsed={got_name!r}", flush=True)
    return {"n": len(results), "n_ok": n_ok, "results": results}


def _probe_one(model: MiniCPM5Audio, case: dict, prompt: str) -> str:
    wav, _ = librosa.load(str(case["wav"]), sr=16000, mono=True)
    return model.generate_text(wav, tools=None, system=prompt)


def _eval_conflict(model: MiniCPM5Audio, cases: list[dict]) -> dict:
    prompts = {
        "asr": ASR_SYSTEM,
        "route_only": ROUTE_ONLY_SYSTEM,
        "reply": REPLY_SYSTEM,
        "mixed": ROUTE_SYSTEM,
    }
    rows = []
    n_route = n_route_ok = 0
    cer_sum = 0.0
    n_cer = 0
    n_parrot = 0
    n_reply = 0
    n_t2t = 0
    t2t_s2t_cer = 0.0
    for case in cases:
        rec: dict = {"id": case["id"], "need_fc": case["need_fc"], "asr": case["asr"]}
        for key, sys_p in prompts.items():
            rec[key] = _probe_one(model, case, sys_p)
        rec["asr_cer"] = cer(case["asr"], rec["asr"])
        if case["asr"]:
            cer_sum += rec["asr_cer"]
            n_cer += 1
        parsed = parse_stream(rec["mixed"])
        rec["mixed_mode"] = parsed["mode"]
        expect_route = bool(case["need_fc"])
        n_route += 1
        got_route = parsed["mode"] == "route" or (
            parsed["mode"] is None and MODE_HINT_ROUTE(rec["route_only"])
        )
        rec["route_pred"] = got_route
        n_route_ok += int(got_route == expect_route)
        reply = rec["reply"]
        rec["parrot"] = is_parrot(reply, case["asr"])
        if not expect_route:
            n_reply += 1
            n_parrot += int(rec["parrot"])
            t2t = model.generate_from_text(case["text"], tools=None, system=REPLY_SYSTEM)
            rec["t2t_reply"] = t2t
            rec["t2t_s2t_cer"] = cer(t2t, reply)
            t2t_s2t_cer += rec["t2t_s2t_cer"]
            n_t2t += 1
        rows.append(rec)
        print(
            f"  conflict {case['id']} asr_cer={rec['asr_cer']:.3f} "
            f"route={got_route}/{expect_route} parrot={rec['parrot']}",
            flush=True,
        )
    return {
        "n": len(rows),
        "asr_cer": (cer_sum / n_cer) if n_cer else None,
        "route_acc": (n_route_ok / n_route) if n_route else None,
        "speak_parrot_rate": (n_parrot / n_reply) if n_reply else None,
        "t2t_s2t_cer": (t2t_s2t_cer / n_t2t) if n_t2t else None,
        "results": rows,
    }


def MODE_HINT_ROUTE(text: str) -> bool:
    t = norm_text(text)
    return "route" in t or "需要" in t or t in ("是", "true", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--n-chat", type=int, default=22)
    ap.add_argument("--max-cases", type=int, default=0, help="cap car+chat each (0=all)")
    ap.add_argument("--skip-text-fc", action="store_true")
    ap.add_argument("--skip-speech-bare", action="store_true")
    ap.add_argument("--skip-speech-fc", action="store_true")
    ap.add_argument("--skip-conflict", action="store_true")
    ap.add_argument("--out-json", default=str(TMP_DIR / "minicpm5_baseline.json"))
    args = ap.parse_args()

    cars = car_wav_cases()
    chats = chat_wav_cases(args.n_chat)
    if args.max_cases:
        cars = cars[: args.max_cases]
        chats = chats[: args.max_cases]
    tools = load_tools()
    report: dict = {"n_car": len(cars), "n_chat": len(chats)}

    if not args.skip_text_fc:
        print("== text FC (bare 2B) ==", flush=True)
        model = _load_model(args, projector=None, adapter=None)
        report["text_fc"] = _eval_text_fc(model, cars, tools)
        del model

    if not args.skip_speech_bare:
        print("== speech FC (align projector, no LoRA) ==", flush=True)
        model = _load_model(args, projector=RUNS_DIR / "align" / "projector.pt", adapter=None)
        report["speech_fc_bare"] = _eval_speech_fc(model, cars, tools)
        if not args.skip_conflict:
            print("== conflict (align projector) ==", flush=True)
            report["conflict_bare"] = _eval_conflict(model, cars + chats)
        del model

    if not args.skip_speech_fc:
        print("== speech FC (v1.0 LoRA) ==", flush=True)
        model = _load_model(
            args,
            projector=RUNS_DIR / "fc" / "projector_final.pt",
            adapter=RUNS_DIR / "fc" / "adapter_final",
        )
        report["speech_fc_lora"] = _eval_speech_fc(model, cars, tools)
        if not args.skip_conflict:
            print("== conflict (v1.0 LoRA) ==", flush=True)
            report["conflict_fc_lora"] = _eval_conflict(model, cars + chats)
        del model

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
