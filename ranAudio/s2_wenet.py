"""Wenet -> S2 NOFC chat screening (does not rewrite wenet_s.jsonl)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ranAudio.protocol import needs_external, norm_text

Q_MARK = ("吗", "呢", "什么", "怎么", "为什么", "是不是", "能不能", "好不好", "你是谁")
DRAMA = (
    "开枪",
    "警官",
    "起义",
    "革命",
    "皇上",
    "共产党",
    "方志敏",
    "铁将军",
    "军阀",
    "八路军",
    "同志",
    "导演",
    "主演",
    "剧组",
    "润之",
    "大哥",
    "崔叔",
    "春明",
    "床头",
    "瞒得了",
    "烦她",
    "被抓",
    "衣服",
    "躲",
    "杀人",
    "死亡",
    "尸体",
    "跪下",
    "枪毙",
)
DIALOGUE_PRON = ("他", "她", "你们", "咱们")
T2T_SYSTEM = (
    "你是语音助手。用一两句中文直接回答用户。"
    "不要复述用户的原话，不要输出工具、JSON 或 XML。"
)
_FN_RE = re.compile(r"<function|</function>|tool_call", re.I)


def ok_src(text: str, duration: float) -> bool:
    t = (text or "").strip()
    if not (1.6 <= duration <= 5.5):
        return False
    if not (8 <= len(t) <= 22):
        return False
    if needs_external(t):
        return False
    if any(k in t for k in DRAMA):
        return False
    if any(p in t for p in DIALOGUE_PRON) and "是什么意思" not in t:
        return False
    if not any(k in t for k in Q_MARK):
        return False
    return True


def ok_reply(user: str, reply: str) -> bool:
    r = (reply or "").strip()
    if not (4 <= len(r) <= 80):
        return False
    if _FN_RE.search(r):
        return False
    if "<|" in r or "<backend>" in r:
        return False
    nu, nr = norm_text(user), norm_text(r)
    if not nr or nu == nr:
        return False
    if nu and nu in nr and len(nu) >= 6:
        return False
    return True


def collect_candidates(src: Path, pool: int, scan: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= scan or len(out) >= pool:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "").strip()
            if text in seen or not ok_src(text, float(row.get("duration") or 0)):
                continue
            seen.add(text)
            out.append(row)
    return out
