#!/usr/bin/env python3
"""Build Step FastLM backend-reinject SFT jsonl.

Input human: <backend>speak</backend> [+ optional Cond audio]
Assistant: <|NOFC|>… (must keep speak facts verbatim)

Sources: FC rows in data/s2_audio_boundary.jsonl + template speaks.
Cond wavs: none / same utterance / short chat / small edge-tts 追问 set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import DATA_DIR, resolve_wav
from ranAudio.protocol import NOFC, wrap_backend
from ranAudio.step_route.protocol_step import REINJECT_SYSTEM

CAR_SPEAK = {
    "enable_bluetooth": "已打开蓝牙。",
    "disable_bluetooth": "已关闭蓝牙。",
    "enable_ac": "空调已打开。",
    "disable_ac": "空调已关闭。",
    "open_window": "车窗已打开。",
    "close_window": "车窗已关上。",
    "volume_up": "音量已调大。",
    "volume_down": "音量已调小。",
    "cancel_navigation": "已取消导航。",
    "answer_call": "已接听来电。",
    "reject_call": "已拒接来电。",
}

FOLLOWUPS = [
    ("嗯", "嗯。"),
    ("haode", "好的。"),
    ("ne", "呢？"),
    ("wendu", "温度呢？"),
    ("xiayu", "会下雨吗？"),
    ("duoshao", "具体多少度？"),
    ("jiage", "现在什么价？"),
    ("haiyou", "还有呢？"),
]


def _set_volume_speak(user: str) -> str:
    m = re.search(r"(\d{1,3})\s*级?|调到\s*(\d{1,3})", user or "")
    n = m.group(1) or m.group(2) if m else None
    if n:
        return f"音量已调到 {n}。"
    return "音量已调整。"


def _set_temp_speak(user: str) -> str:
    m = re.search(r"(\d{1,2})\s*度", user or "")
    if m:
        return f"温度已调到 {m.group(1)} 度。"
    return "温度已调整。"


def _nav_speak(user: str) -> str:
    for key in ("家", "公司", "机场", "车站", "医院"):
        if key in (user or ""):
            return f"已开始导航去{key}。"
    m = re.search(r"导航(?:到|去)?(.{1,12})", user or "")
    if m:
        dest = m.group(1).strip(" 。？?，,")
        if dest:
            return f"已开始导航去{dest}。"
    return "已开始导航。"


def _weather_speak(user: str) -> str:
    city = "北京"
    for c in ("上海", "深圳", "广州", "杭州", "成都", "哈尔滨", "北京"):
        if c in (user or ""):
            city = c
            break
    day = "今天"
    for d in ("后天", "明天", "今天"):
        if d in (user or ""):
            day = d
            break
    # Fixed demo facts; digits must appear in target.
    table = {
        ("北京", "今天"): "北京今天多云，气温 12 到 22 度。",
        ("北京", "明天"): "北京明天多云转晴，18 到 24 度。",
        ("北京", "后天"): "北京后天阴有小雨，10 到 18 度。",
        ("上海", "今天"): "上海今天晴，气温 15 到 25 度。",
        ("上海", "明天"): "上海明天多云，16 到 23 度。",
        ("哈尔滨", "后天"): "哈尔滨后天有雪，气温 零下 12 到 零下 3 度。",
    }
    return table.get((city, day), f"{city}{day}多云，气温 15 到 22 度。")


def _stock_speak(user: str) -> str:
    name = "特斯拉"
    for n in ("苹果", "腾讯", "阿里", "茅台", "特斯拉"):
        if n in (user or ""):
            name = n
            break
    prices = {
        "特斯拉": "178.50",
        "苹果": "226.30",
        "腾讯": "382.40",
        "阿里": "88.60",
        "茅台": "1520.00",
    }
    return f"{name}最新股价为 {prices.get(name, '100.00')} 美元（演示结果）。"


def _search_speak(user: str) -> str:
    return "已为你找到相关结果：今日要闻三条（演示结果）。"


def _time_speak(user: str) -> str:
    if "星期" in (user or "") or "周几" in (user or ""):
        return "今天是星期三。"
    return "现在是下午 3 点 20 分。"


_CN_HOUR = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _alarm_speak(user: str) -> str:
    u = user or ""
    m = re.search(r"(\d{1,2})\s*[点:：]\s*(\d{1,2})?", u)
    if m:
        h = int(m.group(1))
        mi = int(m.group(2) or 0)
        return f"已设置闹钟为 {h} 点 {mi:02d} 分。"
    m = re.search(r"(\d{1,2})\s*点", u)
    if m:
        return f"已设置闹钟为 {m.group(1)} 点整。"
    # 七点 / 七点半 / 十二点
    m = re.search(r"(十一|十二|十|[零一二两三四五六七八九])\s*点\s*(半)?", u)
    if m:
        h = _CN_HOUR.get(m.group(1), 7)
        if m.group(2):
            return f"已设置闹钟为 {h} 点 30 分。"
        return f"已设置闹钟为 {h} 点整。"
    return "已设置闹钟。"


def _remind_speak(user: str) -> str:
    u = user or ""
    when = ""
    for w in ("下午", "早上", "上午", "晚上", "明天", "后天"):
        if w in u:
            when = w
            break
    # strip common verbs to leave a short topic
    topic = u
    for p in ("提醒我", "帮我提醒", "叫我", "记得", "谢谢", "请"):
        topic = topic.replace(p, "")
    topic = topic.strip(" ，,。！!？?")
    if when and topic:
        return f"好的，到{when}会提醒你：{topic}。"
    if topic:
        return f"好的，已设置提醒：{topic}。"
    return "好的，已设置提醒。"


def _order_speak(user: str) -> str:
    u = user or ""
    m = re.search(r"(下单|订|点)(一份|个|一碗|一盘)?(.{1,12})", u)
    if m:
        item = m.group(3).strip(" 。？?，,")
        if item:
            return f"已为你下单：{item}。"
    return "已提交订单。"


def build_speak(row: dict) -> str:
    """Simulate SlowLM/tool result: must be request-specific, not a constant OK."""
    tool = row.get("tool") or ""
    user = row.get("user") or ""
    if tool in CAR_SPEAK:
        return CAR_SPEAK[tool]
    if tool == "set_volume":
        return _set_volume_speak(user)
    if tool == "set_temperature":
        return _set_temp_speak(user)
    if tool == "navigate_to":
        return _nav_speak(user)
    # toolish / free text — match by intent keywords
    if any(k in user for k in ("天气", "下雨", "下雪", "气温")):
        return _weather_speak(user)
    if any(k in user for k in ("股价", "股票", "特斯拉", "苹果股价")):
        return _stock_speak(user)
    if any(k in user for k in ("几点", "时间", "星期", "日期", "几号", "现在几")):
        return _time_speak(user)
    if any(k in user for k in ("闹钟", "定个时", "定闹钟", "设个闹钟")):
        return _alarm_speak(user)
    if any(k in user for k in ("提醒", "叫我", "记得")):
        return _remind_speak(user)
    if any(k in user for k in ("下单", "订餐", "订一份", "点一份", "点个")):
        return _order_speak(user)
    if any(k in user for k in ("搜", "查一下", "新闻", "路况", "搜索")):
        return _search_speak(user)
    if "蓝牙" in user:
        return "已打开蓝牙。" if ("打开" in user or "开" in user) and "关" not in user[:2] else (
            "已关闭蓝牙。" if "关" in user else "已打开蓝牙。"
        )
    if "空调" in user:
        return "空调已打开。" if "开" in user and "关" not in user else (
            "空调已关闭。" if "关" in user else "空调已打开。"
        )
    if "导航" in user:
        return _nav_speak(user)
    if "音量" in user:
        return _set_volume_speak(user)
    if "温度" in user or re.search(r"调到\s*\d+\s*度", user):
        return _set_temp_speak(user)
    # Last resort: still bind to user text so facts differ across samples.
    brief = (user or "你的请求").strip(" 。？?！!")
    if len(brief) > 24:
        brief = brief[:24]
    return f"已处理：{brief}。"


def build_target(speak: str, rng: random.Random) -> str:
    """Prefix is only glue; backend speak must appear verbatim in the target."""
    speak = speak.strip()
    # Query-style facts can use 查到了；确认/提醒类不要。
    is_confirm = speak.startswith(("已", "好的", "音量已", "温度已", "空调已"))
    if is_confirm:
        prefixes = ["", "对，"]
        weights = [70, 30]
    else:
        prefixes = ["", "查到了，", "好的，", "对，"]
        weights = [40, 25, 20, 15]
    prefix = rng.choices(prefixes, weights=weights, k=1)[0]
    if prefix and speak.startswith(prefix.rstrip("，")):
        prefix = ""
    if prefix == "好的，" and speak.startswith("好的"):
        prefix = ""
    return f"{NOFC}{prefix}{speak}"


def _synth_followups_qwen(out_dir: Path, missing: list[tuple[str, str]]) -> None:
    """Local Qwen3-TTS CustomVoice fallback when edge-tts is blocked."""
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model_path = Path("/cmy/cmy/Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    print(f"[cond] qwen-tts synthesize {len(missing)} followups…", flush=True)
    model = Qwen3TTSModel.from_pretrained(str(model_path), device_map="cuda:0")
    for key, text in missing:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker="Vivian",
            language="Chinese",
            non_streaming_mode=True,
        )
        audio = np.asarray(wavs[0], dtype=np.float32)
        path = out_dir / f"cond_{key}.wav"
        sf.write(str(path), audio, int(sr))
        print(f"[cond] wrote {path.name} sr={sr} n={audio.shape[0]}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def ensure_followup_wavs(out_dir: Path, voice: str = "zh-CN-XiaoxiaoNeural") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    missing: list[tuple[str, str]] = []
    for key, text in FOLLOWUPS:
        path = out_dir / f"cond_{key}.wav"
        paths[key] = path
        if path.is_file() and path.stat().st_size > 1000:
            continue
        missing.append((key, text))
    if not missing:
        return paths

    # Prefer edge-tts; on failure fall back to local Qwen3-TTS.
    try:
        import edge_tts

        for key, text in missing:
            path = paths[key]
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(path))
            if not path.is_file() or path.stat().st_size < 1000:
                raise RuntimeError(f"edge-tts empty: {path}")
        return paths
    except Exception as e:
        print(f"[cond] edge-tts failed ({type(e).__name__}: {e}); try qwen-tts", flush=True)

    still = [(k, t) for k, t in missing if not (paths[k].is_file() and paths[k].stat().st_size > 1000)]
    if still:
        _synth_followups_qwen(out_dir, still)
    return paths


def _list_chat_wavs() -> list[Path]:
    d = DATA_DIR / "chat_wavs"
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.wav") if p.stat().st_size > 1000)


def make_human_content(backend: str, cond_wav: Path | None):
    if cond_wav is None:
        return backend
    return [
        {"type": "text", "text": backend + "\n"},
        {"type": "audio", "audio": str(cond_wav.resolve())},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DATA_DIR / "s2_audio_boundary.jsonl")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "step_s2_backend.jsonl")
    ap.add_argument("--cond-dir", type=Path, default=DATA_DIR / "step_backend_cond_wavs")
    ap.add_argument("--max-base", type=int, default=2800, help="max FC/toolish rows to expand")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-tts", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    followups: dict[str, Path] = {}
    if not args.skip_tts:
        followups = asyncio.run(ensure_followup_wavs(args.cond_dir))
        print(f"[cond] followup wavs={len(followups)} dir={args.cond_dir}", flush=True)

    chat_wavs = _list_chat_wavs()
    print(f"[cond] chat_wavs={len(chat_wavs)}", flush=True)

    bases: list[dict] = []
    with args.src.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("route") != "fc":
                continue
            wav = resolve_wav(row.get("wav") or "")
            if not wav.is_file():
                continue
            bases.append({**row, "_wav": wav})

    rng.shuffle(bases)
    bases = bases[: max(1, args.max_base)]

    # Cond type mix per base row (one sample each draw to control size).
    # none 35%, same 25%, chat 20%, followup 20%
    out_rows: list[dict] = []
    for i, row in enumerate(bases):
        speak = build_speak(row)
        target = build_target(speak, rng)
        backend = wrap_backend(speak)
        roll = rng.random()
        cond_wav: Path | None
        cond_kind: str
        if followups:
            # none 30% / same 25% / chat 20% / followup 25%
            if roll < 0.30:
                cond_wav, cond_kind = None, "none"
            elif roll < 0.55:
                cond_wav, cond_kind = row["_wav"], "same"
            elif roll < 0.75 and chat_wavs:
                cond_wav, cond_kind = rng.choice(chat_wavs), "chat"
            else:
                key = rng.choice(list(followups.keys()))
                if "度" in speak or "天气" in speak or "雨" in speak or "雪" in speak:
                    key = rng.choice([k for k in ("wendu", "xiayu", "duoshao", "嗯", "haode") if k in followups])
                elif "股价" in speak or "美元" in speak:
                    key = rng.choice([k for k in ("jiage", "haiyou", "haode", "ne") if k in followups])
                cond_wav, cond_kind = followups[key], f"followup:{key}"
        else:
            # no TTS: none 40% / same 35% / chat 25%
            if roll < 0.40:
                cond_wav, cond_kind = None, "none"
            elif roll < 0.75:
                cond_wav, cond_kind = row["_wav"], "same"
            elif chat_wavs:
                cond_wav, cond_kind = rng.choice(chat_wavs), "chat"
            else:
                cond_wav, cond_kind = None, "none"

        human = make_human_content(backend, cond_wav)
        messages = [
            {"role": "system", "content": REINJECT_SYSTEM},
            {"role": "human", "content": human},
            {"role": "assistant", "content": target},
        ]
        # Sanity: speak substring in target
        if speak not in target:
            raise RuntimeError(f"fact missing in target: {speak!r} vs {target!r}")

        out_rows.append(
            {
                "id": f"backend_{i:05d}_{row.get('id')}",
                "phase": "backend",
                "route": "nofc",
                "kind": "backend",
                "cond_kind": cond_kind,
                "user": row.get("user"),
                "speak": speak,
                "target": target,
                "wav": str(cond_wav.resolve()) if cond_wav else None,
                "src_wav": str(row["_wav"].resolve()),
                "tool": row.get("tool"),
                "source": row.get("source"),
                "messages": messages,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    print(
        json.dumps(
            {
                "out": str(args.out),
                "n": len(out_rows),
                "cond": dict(Counter(r["cond_kind"].split(":")[0] for r in out_rows)),
                "followup_detail": dict(
                    Counter(r["cond_kind"] for r in out_rows if r["cond_kind"].startswith("followup"))
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
