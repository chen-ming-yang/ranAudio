#!/usr/bin/env python3
"""Held-out S2 test set: new FC / chat / asr lines, edge-tts 16 kHz.

Drops any user string already present in data/s2_audio.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.build_judge_samples import default_chat_reply
from ranAudio.gen_s2_tts import run_edge, wav_ok
from ranAudio.paths import CAR_TEXTS, CAR_WAVS, S2_AUDIO_JSONL, S2_TEST_JSONL, S2_TEST_WAVS, rel_wav
from ranAudio.protocol import format_fc_base, format_nofc, needs_external
from ranAudio.s2_catalog import ASR_TRIGGERS, EDGE_VOICES, read_jsonl, write_jsonl

FC_POOL = [
    "帮我看看哈尔滨后天会不会下雪",
    "查一下大理今天的紫外线",
    "现在东京几点",
    "帮我搜一下附近的宠物医院",
    "茅台今天收盘价是多少",
    "把副驾座椅加热打开",
    "导航去最近的充电站",
    "取消刚才定的那个闹钟",
    "帮我查顺丰快递到哪了",
    "把音量调到十五",
    "把天窗打开一条缝",
    "查一下明天北京到成都的高铁还有票吗",
    "三亚这周末台风吗",
    "帮我看看宁波明天气温",
    "搜一下附近二十四小时药店",
    "苹果股票现在多少钱",
    "定个闹钟后天早上六点四十",
    "帮我下单一份牛肉面",
    "去虹桥机场堵不堵",
    "把后排空调温度调到二十三度",
    "关闭座舱氛围灯",
    "帮我查一下明天有没有雨",
    "附近有没有洗车的地方",
    "今天上证指数怎么样",
    "提醒我晚上九点给妈妈打电话",
    "导航去最近的医院",
    "把主驾车窗开一半",
    "查一下国航CA1234延误了没有",
    "昆明今天空气质量怎么样",
]

CHAT_POOL = [
    ("香蕉用英语怎么说", "Banana。还想翻别的词也可以说。"),
    ("背一首春晓", "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"),
    ("三乘七等于几", "三乘七等于二十一。"),
    ("来个脑筋急转弯", "什么东西越洗越脏？水。"),
    ("我有点困了", "先闭眼歇两分钟。想听笑话也可以说。"),
    ("你是男的还是女的", "我没有性别，就是这边的语音助手。"),
    ("把谢谢翻译成日语", "日语里谢谢常说ありがとう。"),
    ("什么是闰年", "公历里多一天的年份，通常每四年一次，二月有二十九天。"),
    ("来段绕口令听听", "四是四，十是十，十四是十四，四十是四十。"),
    ("讲个冷知识", "蜂蜜不会坏，考古里出土过还能吃的蜂蜜。"),
    ("我今天有点紧张", "先慢慢呼气。紧张的时候把要说的话缩成三句就好。"),
    ("你会不会下棋", "规则我可以聊。真要和你下一盘，这边没有棋盘。"),
    ("推荐一个睡前习惯", "睡前把手机放到够不着的地方，灯调暗，固定一个上床时间。"),
    ("什么是守株待兔的意思", "守株待兔是把一次碰巧当成以后还会发生，干等着。"),
    ("十二的一半是多少", "十二的一半是六。"),
    ("用英文说早上好", "Good morning。"),
    ("给我一句鼓励的话", "你已经在做了。先完成最小的那一步。"),
    ("什么动物跑得最快", "陆地上一般说猎豹最快。"),
    ("来首小诗就两句", "白日依山尽，黄河入海流。"),
    ("我该怎么称呼你", "叫我语音助手就行。"),
]

ASR_BODIES = [
    "会议室的投影仪坏了",
    "把合同寄到上海分公司",
    "周五前把报价单发出去",
    "访客停在地库B3",
    "样品放到二号冷柜",
    "把会议改到周四上午",
    "发票抬头写成星河科技",
    "备用钥匙放在前台抽屉",
    "孩子三点在校门口等",
    "把旧工牌交还给行政",
    "今晚列车晚点四十分钟",
    "把桌牌按姓氏拼音排",
    "药房的门禁卡失效了",
    "把录音只保留内部版",
    "下周一开始远程办公",
    "把海报贴在电梯左侧",
    "客户姓周不姓邹",
    "把试驾安排在周六",
]

ASR_TRAPS = [
    "打开天窗",
    "关闭氛围灯",
    "查一下沪深三百",
    "帮我订一张去厦门的机票",
    "把主驾座椅往前调",
    "搜索附近的充电桩",
]


def _train_users(path: Path) -> set[str]:
    return {str(r.get("user") or "").strip() for r in read_jsonl(path)}


def _take(pool: list, n: int, seen: set[str]) -> list:
    out = []
    for item in pool:
        text = item[0] if isinstance(item, tuple) else item
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(item if isinstance(item, tuple) else text)
        if len(out) >= n:
            break
    return out


def build_rows(train_users: set[str], n_fc: int, n_chat: int, n_asr: int, n_trap: int) -> list[dict]:
    seen = set(train_users)
    rows: list[dict] = []
    for i, text in enumerate(_take(FC_POOL, n_fc, seen)):
        rows.append(
            {
                "id": f"s2t_fc_{i:03d}",
                "bucket": "fc",
                "route": "fc",
                "venue": "base",
                "user": text,
                "target": format_fc_base(text),
                "engine": "edge",
                "voice": EDGE_VOICES[i % len(EDGE_VOICES)],
                "source": "s2_test",
            }
        )
    for i, (user, reply) in enumerate(_take(CHAT_POOL, n_chat, seen)):
        if needs_external(user):
            continue
        rows.append(
            {
                "id": f"s2t_chat_{i:03d}",
                "bucket": "chat",
                "route": "nofc",
                "kind": "chat",
                "user": user,
                "reply": reply or default_chat_reply(user),
                "target": format_nofc(reply or default_chat_reply(user)),
                "engine": "edge",
                "voice": EDGE_VOICES[i % len(EDGE_VOICES)],
                "source": "s2_test",
            }
        )
    bodies = _take(ASR_BODIES, n_asr - n_trap, seen)
    traps = _take(ASR_TRAPS, n_trap, seen)
    asr_i = 0
    for body, trap in [(b, False) for b in bodies] + [(b, True) for b in traps]:
        trigger = ASR_TRIGGERS[asr_i % len(ASR_TRIGGERS)]
        user = f"{trigger}：{body}"
        if user in seen:
            user = f"{trigger}，{body}"
        if user in seen:
            continue
        seen.add(user)
        rows.append(
            {
                "id": f"s2t_asr_{asr_i:03d}",
                "bucket": "asr",
                "route": "nofc",
                "kind": "asr",
                "user": user,
                "body": body,
                "trap": trap,
                "target": format_nofc(body),
                "engine": "edge",
                "voice": EDGE_VOICES[asr_i % len(EDGE_VOICES)],
                "source": "s2_test",
            }
        )
        asr_i += 1
    return rows


def car_holdout_rows() -> list[dict]:
    out = []
    for row in read_jsonl(CAR_TEXTS):
        sid = str(row.get("id") or "")
        if not sid.endswith("_000"):
            continue
        wav = CAR_WAVS / f"{sid}.wav"
        if not wav.is_file():
            continue
        text = str(row.get("text") or "").strip()
        out.append(
            {
                "id": f"holdout_{sid}",
                "wav": rel_wav(wav),
                "bucket": "fc",
                "route": "fc",
                "venue": "base",
                "user": text,
                "target": format_fc_base(text),
                "source": "car_holdout",
                "engine": "edge-tts",
                "voice": row.get("voice"),
                "tool": row.get("tool"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(S2_AUDIO_JSONL))
    ap.add_argument("--out", default=str(S2_TEST_JSONL))
    ap.add_argument("--wav-dir", default=str(S2_TEST_WAVS))
    ap.add_argument("--n-fc", type=int, default=24)
    ap.add_argument("--n-chat", type=int, default=16)
    ap.add_argument("--n-asr", type=int, default=20)
    ap.add_argument("--n-trap", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--skip-tts", action="store_true")
    args = ap.parse_args()

    users = _train_users(Path(args.train))
    fresh = build_rows(users, args.n_fc, args.n_chat, args.n_asr, args.n_trap)
    wav_dir = Path(args.wav_dir)
    if not args.skip_tts:
        asyncio.run(run_edge(fresh, wav_dir, args.concurrency))
    kept = []
    for row in fresh:
        wav = wav_dir / f"{row['id']}.wav"
        if not wav_ok(wav):
            print(f"drop missing wav {row['id']}", flush=True)
            continue
        rec = dict(row)
        rec["wav"] = rel_wav(wav)
        kept.append(rec)
    held = car_holdout_rows()
    rows = kept + held
    n = write_jsonl(Path(args.out), rows)
    print(
        f"wrote {n} test rows fresh={len(kept)} holdout={len(held)} -> {args.out}",
        flush=True,
    )
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
