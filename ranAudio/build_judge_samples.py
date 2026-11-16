#!/usr/bin/env python3
"""Build judge jsonl: car need_fc=true (asr only) + chat need_fc=false (asr+reply)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import (
    CAR_TEXTS,
    CAR_WAVS,
    CHAT_TEXTS,
    CHAT_WAVS,
    JUDGE_JSONL,
    rel_wav,
)


def default_chat_reply(text: str) -> str:
    """Short template reply so 2B learns to act as a normal LLM when need_fc=false."""
    t = (text or "").strip()
    if not t:
        return "我在，有什么想聊的吗？"
    # Lightweight patterned replies for common chat intents in chat_texts.jsonl
    if any(k in t for k in ("笑话", "故事", "诗", "相声")):
        return "好啊。从前有个人走路太认真，结果把路走完了——就这，哈哈。"
    if any(k in t for k in ("你是谁", "介绍一下你", "你会做什么")):
        return "我是语音助手。闲聊和问答我可以直接答；要查实时信息或控设备时我会转出去执行。"
    if any(k in t for k in ("你好", "早上好", "晚安", "在吗", "听到")):
        return "你好，我在。想聊天直接说就行。"
    if any(k in t for k in ("谢谢", "不客气", "收到", "明白", "知道了")):
        return "好的。"
    if any(k in t for k in ("翻译", "英语", "hello")):
        return "Hello 可以译成「你好」。还想翻哪句跟我说。"
    if any(k in t for k in ("一加一", "等于几", "乘法", "随机数")):
        return "一加一等于二。还想算别的也可以说。"
    if any(k in t for k in ("电影", "剧", "歌", "推荐")):
        return "我可以聊聊口味。真要播放或下单再跟我说具体指令。"
    return f"好的，关于「{t}」——我可以陪你聊聊。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--car-texts", default=str(CAR_TEXTS))
    ap.add_argument("--car-wav-dir", default=str(CAR_WAVS))
    ap.add_argument("--chat-texts", default=str(CHAT_TEXTS))
    ap.add_argument("--chat-wav-dir", default=str(CHAT_WAVS))
    ap.add_argument("--out", default=str(JUDGE_JSONL))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = missing = 0

    with out.open("w", encoding="utf-8") as w:
        car_wav_dir = Path(args.car_wav_dir)
        with Path(args.car_texts).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("id") or "")
                wav = car_wav_dir / f"{sid}.wav"
                if not wav.is_file():
                    missing += 1
                    continue
                text = str(row.get("text") or "")
                rec = {
                    "id": sid,
                    "wav": rel_wav(wav),
                    "asr": text,
                    "text": text,
                    "need_fc": True,
                    "tool": str(row.get("tool") or ""),
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pos += 1

        chat_wav_dir = Path(args.chat_wav_dir)
        chat_texts = Path(args.chat_texts)
        if chat_texts.is_file():
            with chat_texts.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    sid = str(row.get("id") or "")
                    wav = chat_wav_dir / f"{sid}.wav"
                    if not wav.is_file():
                        missing += 1
                        continue
                    text = str(row.get("text") or "")
                    reply = str(row.get("reply") or default_chat_reply(text))
                    rec = {
                        "id": sid,
                        "wav": rel_wav(wav),
                        "asr": text,
                        "text": text,
                        "need_fc": False,
                        "reply": reply,
                        "tool": "",
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_neg += 1

    print(
        f"wrote judge pos={n_pos} neg={n_neg} total={n_pos + n_neg} -> {out} missing_wav={missing}",
        flush=True,
    )
    if n_neg == 0:
        print("warn: no negative chat wavs; run gen_chat_tts first", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
