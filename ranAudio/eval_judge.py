#!/usr/bin/env python3
"""Judge eval: need_fc + ASR; negatives also require non-empty reply."""

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

from ranAudio.model import MiniCPM5Audio
from ranAudio.paths import CAR_WAVS, CHAT_WAVS, JUDGE_JSONL, MINICPM5, MINICPMO, RUNS_DIR, TMP_DIR
from ranAudio.tools import judge_system_prompt, parse_judge

TEST_TOOLS = [
    "enable_bluetooth",
    "disable_bluetooth",
    "navigate_to",
    "cancel_navigation",
    "enable_ac",
    "disable_ac",
    "set_temperature",
    "volume_up",
    "volume_down",
    "set_volume",
    "open_window",
    "close_window",
    "answer_call",
    "reject_call",
    "play_music",
    "pause_music",
    "next_track",
    "previous_track",
    "enable_seat_heating",
    "disable_seat_heating",
    "enable_headlights",
    "disable_headlights",
]


def _norm(s: str) -> str:
    return "".join(str(s or "").split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=str(MINICPM5))
    ap.add_argument("--apm", default=str(MINICPMO))
    ap.add_argument("--projector", default=str(RUNS_DIR / "judge" / "projector_final.pt"))
    ap.add_argument("--adapter", default=str(RUNS_DIR / "judge" / "adapter_final"))
    ap.add_argument("--samples", default=str(JUDGE_JSONL))
    ap.add_argument("--n-neg", type=int, default=22)
    ap.add_argument("--out-json", default=str(TMP_DIR / "minicpm5_judge_eval.json"))
    args = ap.parse_args()

    by_id: dict[str, dict] = {}
    if Path(args.samples).is_file():
        with Path(args.samples).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_id[str(row.get("id") or "")] = row

    # (wav, expect_fc, expect_asr)
    cases: list[tuple[Path, bool, str]] = []
    for name in TEST_TOOLS:
        wav = CAR_WAVS / f"{name}_000.wav"
        if not wav.is_file():
            print(f"warn missing {wav}", flush=True)
            continue
        row = by_id.get(wav.stem) or {}
        asr = str(row.get("asr") or row.get("text") or "")
        cases.append((wav, True, asr))

    neg_ids = sorted(p.stem for p in CHAT_WAVS.glob("chat_*.wav"))[: max(0, args.n_neg)]
    for sid in neg_ids:
        wav = CHAT_WAVS / f"{sid}.wav"
        row = by_id.get(sid) or {}
        asr = str(row.get("asr") or row.get("text") or "")
        cases.append((wav, False, asr))

    model = MiniCPM5Audio.from_pretrained(
        args.llm,
        args.apm,
        projector_path=args.projector if Path(args.projector).is_file() else None,
        encoder_backend="whisper",
    )
    if Path(args.adapter).is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, args.adapter)
        print(f"loaded adapter {args.adapter}", flush=True)
    model.eval()
    system = judge_system_prompt()

    results = []
    n_fc_ok = n_asr_ok = n_reply_ok = n_both = 0
    for wav_path, expect_fc, expect_asr in cases:
        wav, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        text = model.generate_text(wav, tools=None, system=system)
        parsed = parse_judge(text) or {}
        got_fc = parsed.get("need_fc")
        got_asr = str(parsed.get("asr") or "")
        got_reply = str(parsed.get("reply") or "")
        fc_ok = got_fc is expect_fc if got_fc is not None else False
        asr_ok = _norm(got_asr) == _norm(expect_asr) and bool(expect_asr)
        # need_fc=true: must NOT rely on reply; need_fc=false: reply must be non-empty
        if expect_fc:
            reply_ok = True
        else:
            reply_ok = bool(got_reply.strip())
        both = fc_ok and asr_ok and reply_ok
        if fc_ok:
            n_fc_ok += 1
        if asr_ok:
            n_asr_ok += 1
        if reply_ok:
            n_reply_ok += 1
        if both:
            n_both += 1
        tag = "PASS" if both else "FAIL"
        rec = {
            "wav": wav_path.name,
            "expect_need_fc": expect_fc,
            "expect_asr": expect_asr,
            "got_need_fc": got_fc,
            "got_asr": got_asr,
            "got_reply": got_reply if not expect_fc else None,
            "fc_ok": fc_ok,
            "asr_ok": asr_ok,
            "reply_ok": reply_ok,
            "ok": both,
            "text": text,
        }
        results.append(rec)
        print(
            f"  {tag} {wav_path.name} fc={got_fc!r}/{expect_fc} "
            f"asr={got_asr!r}/{expect_asr!r} reply={got_reply!r}",
            flush=True,
        )

    n = len(results)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": n,
        "n_fc_ok": n_fc_ok,
        "n_asr_ok": n_asr_ok,
        "n_reply_ok": n_reply_ok,
        "n_both": n_both,
        "results": results,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"wrote {out}  both={n_both}/{n} need_fc={n_fc_ok}/{n} "
        f"asr={n_asr_ok}/{n} reply={n_reply_ok}/{n}",
        flush=True,
    )
    return 0 if n and n_both == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
