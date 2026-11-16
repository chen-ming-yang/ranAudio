#!/usr/bin/env python3
"""Shared helpers for Kimi voice-edit (demo + eval)."""
from __future__ import annotations

import re

_PUNCT = "。！？!?，、；;：:…·《》〈〉「」『』""''\"'（）()【】[]—–-·、"


def norm_text(s: str) -> str:
    """Whitespace + punctuation stripped; used for exact / CER / echo."""
    t = "".join(str(s or "").split())
    return "".join(ch for ch in t if ch not in _PUNCT)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r = norm_text(ref)
    h = norm_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


_CN_DIG = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "幺": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_NUM = r"[零〇一二三四五六七八九两幺十百千万亿]+"
_PROTECT_RE = re.compile(rf"[{''.join(_CN_DIG)}]点(?:水|旁|头|底|边|框)")
# 千万/万万 as intensifier adverb, not the number 10_000_000
_INTENSIFIER_NUM_RE = re.compile(
    r"(?:千千万万|千万|万万)(?=(?:不要|别|不能|不可|不得|要|得|记住|记得|小心|注意|别忘|别再|别去|别把))"
)
_SYMBOL_WORDS = (
    ("百分号", "%"),
    ("省略号", "…"),
    ("反斜杠", "\\"),
    ("下划线", "_"),
    ("加号", "+"),
    ("减号", "-"),
    ("乘号", "×"),
    ("除号", "÷"),
    ("等号", "="),
    ("井号", "#"),
    ("艾特", "@"),
    ("斜杠", "/"),
    ("星号", "*"),
    ("冒号", ":"),
    ("问号", "?"),
    ("感叹号", "!"),
    ("和号", "&"),
)


def _parse_cn_section(s: str) -> int:
    if not s:
        return 0
    if all(c in _CN_DIG for c in s):
        if len(s) == 1:
            return _CN_DIG[s]
        return int("".join(str(_CN_DIG[c]) for c in s))
    total = 0
    num = 0
    for c in s:
        if c in _CN_DIG:
            num = _CN_DIG[c]
        elif c in _CN_UNIT:
            if num == 0:
                num = 1
            total += num * _CN_UNIT[c]
            num = 0
        else:
            raise ValueError(s)
    return total + num


def _parse_cn(s: str) -> int:
    if "亿" in s:
        left, _, right = s.partition("亿")
        return _parse_cn(left or "一") * 100_000_000 + (_parse_cn(right) if right else 0)
    if "万" in s:
        left, _, right = s.partition("万")
        return _parse_cn(left or "一") * 10_000 + (_parse_cn(right) if right else 0)
    return _parse_cn_section(s)


def _cn_to_str(s: str) -> str:
    if all(c in _CN_DIG for c in s):
        return "".join(str(_CN_DIG[c]) for c in s)
    return str(_parse_cn(s))


def _convert_cn_spans(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        try:
            return _cn_to_str(m.group(0))
        except ValueError:
            return m.group(0)

    return re.sub(_CN_NUM, repl, text)


def normalize_display(text: str) -> str:
    """Chinese numerals, spoken math, and symbol words → digits and symbols."""
    if not text:
        return text or ""
    holds: list[str] = []

    def _hold(m: re.Match[str]) -> str:
        holds.append(m.group(0))
        return f"\ue000{len(holds) - 1}\ue001"

    t = _PROTECT_RE.sub(_hold, text)
    t = _INTENSIFIER_NUM_RE.sub(_hold, t)

    def _percent(m: re.Match[str]) -> str:
        return _cn_to_str(m.group(1)) + "%"

    t = re.sub(rf"百分之({_CN_NUM})", _percent, t)

    def _clock(m: re.Match[str]) -> str:
        hour = _cn_to_str(m.group(1))
        minute = int(_cn_to_str(m.group(2)))
        return f"{hour}:{minute:02d}"

    t = re.sub(rf"({_CN_NUM})点({_CN_NUM})分", _clock, t)

    def _decimal(m: re.Match[str]) -> str:
        frac = m.group(2)
        if any(c in "十百千万亿" for c in frac):
            return m.group(0)
        return _cn_to_str(m.group(1)) + "." + "".join(str(_CN_DIG[c]) for c in frac)

    t = re.sub(rf"({_CN_NUM})点({_CN_NUM})", _decimal, t)

    def _arith(m: re.Match[str]) -> str:
        op = {"加": "+", "减": "-", "乘": "×", "乘以": "×", "除": "÷", "除以": "÷"}[m.group(2)]
        left = f"{_cn_to_str(m.group(1))} {op} {_cn_to_str(m.group(3))}"
        if m.group(4):
            return f"{left} = {_cn_to_str(m.group(4))}"
        return left

    t = re.sub(
        rf"({_CN_NUM})(乘以|除以|加|减|乘|除)({_CN_NUM})(?:等于({_CN_NUM}))?",
        _arith,
        t,
    )
    t = _convert_cn_spans(t)
    for word, sym in _SYMBOL_WORDS:
        t = t.replace(word, sym)
    for i, raw in enumerate(holds):
        t = t.replace(f"\ue000{i}\ue001", raw)
    return t


_PAIR_RE = re.compile(
    r"(?:把|将)(.+?)(?:[，、]*)(?:变成|改成|换成)(.+?)(?=(?:，|、)?(?:把|将)|$|[。！？!?])"
)


def extract_replace_pairs(instruction: str, *, keep_equal: bool = False) -> list[tuple[str, str]]:
    """Pull all 把/将 A 变成|改成|换成 B clauses from an instruction."""
    instr = (instruction or "").strip()
    instr = re.sub(r"[。！？!?]+$", "", instr).strip()
    if not instr:
        return []
    pairs: list[tuple[str, str]] = []
    for m in _PAIR_RE.finditer(instr):
        src, dst = m.group(1).strip(), m.group(2).strip()
        src = re.sub(r"[，、。！？!?]+$", "", src).strip()
        dst = re.sub(r"[吧啊呀呢，、。！？!?]+$", "", dst).strip()
        if src and dst and (keep_equal or src != dst):
            pairs.append((src, dst))
    return pairs


def apply_simple_replace(draft: str, instruction: str) -> str | None:
    """Handle spoken edits like 把A改成B, including multi-clause instructions."""
    instr = (instruction or "").strip()
    if not instr or not draft:
        return None

    pairs = extract_replace_pairs(instr)
    if pairs:
        # longer src first so 明天下午 beats 明天
        pairs = sorted(pairs, key=lambda x: -len(x[0]))
        out = draft
        hits = 0
        for src, dst in pairs:
            if src in out:
                out = out.replace(src, dst, 1)
                hits += 1
        if hits and out != draft:
            return out
        return None

    # single-clause fallback (no 把/将), fullmatch only
    patterns = [
        r"把(.+?)变成(.+)",
        r"把(.+?)改成(.+)",
        r"将(.+?)变成(.+)",
        r"将(.+?)改成(.+)",
        r"把(.+?)换成(.+)",
        r"(.+?)改成(.+)",
        r"(.+?)换成(.+)",
    ]
    instr_one = re.sub(r"[。！？!?]+$", "", instr).strip()
    for pat in patterns:
        m = re.fullmatch(pat, instr_one)
        if not m:
            continue
        src, dst = m.group(1).strip(), m.group(2).strip()
        dst = re.sub(r"[吧啊呀呢。！？!?]+$", "", dst).strip()
        if src and dst and src != dst and src in draft:
            return draft.replace(src, dst, 1)
    return None


_VERB = r"(?:修改为|改成|换成|变成|改作)"
_BOTH_RADICAL_RE = re.compile(
    rf"^把(.{{1,8}}?)的(.){_VERB}(.{{1,8}}?)的(.)$"
)
_TARGET_RADICAL_RE = re.compile(rf"^把(.){_VERB}(.{{1,8}}?)的(.)$")
_SPLIT_CHAR_RE = re.compile(rf"^把(.){_VERB}([^的]{{2,4}})(.)$")


def _swap_once(draft: str, src: str, dst: str) -> str | None:
    if not src or not dst or src == dst or src not in draft:
        return None
    return draft.replace(src, dst, 1)


def apply_radical_describe_replace(draft: str, instruction: str) -> str | None:
    """Glyph edits described by radicals, e.g. 把白字旁的的修改为提土旁的地.

    Does not handle 'describe source only' (把女字旁的她改成他).
    """
    if not draft or not instruction:
        return None
    instr = re.sub(r"[。！？!?]+$", "", (instruction or "").strip()).strip()
    m = _BOTH_RADICAL_RE.fullmatch(instr)
    if m:
        return _swap_once(draft, m.group(2), m.group(4))
    m = _TARGET_RADICAL_RE.fullmatch(instr)
    if m:
        return _swap_once(draft, m.group(1), m.group(3))
    m = _SPLIT_CHAR_RE.fullmatch(instr)
    if m:
        return _swap_once(draft, m.group(1), m.group(3))
    return None


# Inline ASR name clarifications: 「名是名字的名」「杨是提手旁的杨」
_CLARIFY_CLAUSE_RE = re.compile(r"^(.)是(.+?)的(.)$")
_NAME_HEAD_RE = re.compile(r"^(我叫|我是)(.+)$")
_RADICAL_DESC_RE = re.compile(
    r"(?:旁|头|底|边|框|盖)$|三点水|提手旁|提土旁|木字旁|言字旁|"
    r"女字旁|单人旁|立人旁|双人旁|宝盖头|日字旁|口字旁|门字框|"
    r"绞丝旁|王字旁|走之底|竹字头|穴字头|工字旁|土字旁|土字底|青字底|心字底"
)
# ASR often mangles radical names; map to canonical keys used below.
_RADICAL_ALIASES: dict[str, str] = {
    "铁手旁": "提手旁",
    "剔手旁": "提手旁",
    "体手旁": "提手旁",
    "提首旁": "提手旁",
    "提士旁": "提土旁",
    "剔土旁": "提土旁",
    "铁土旁": "提土旁",
    "三点谁": "三点水",
    "三点谁旁": "三点水",
    "言子旁": "言字旁",
    "单人旁旁": "单人旁",
}

# Extra spoken surfaces that ASR confuses for a radical's true glyph.
# e.g. 「提手旁的扬」→「铁手旁的羊」：羊 is the phonetic of 扬.
_RADICAL_SURFACE_ALIAS: dict[str, dict[str, str]] = {
    "提手旁": {"羊": "扬", "阳": "扬", "杨": "扬", "扬": "扬"},
    "木字旁": {"扬": "杨", "羊": "杨", "阳": "杨", "杨": "杨"},
    "三点水": {"清": "清", "青": "清", "请": "清"},
    "言字旁": {"请": "请", "青": "请", "清": "请"},
}

# (radical_desc, spoken_surface) → correct glyph
_RADICAL_CHAR_FIX: dict[tuple[str, str], str] = {
    ("提手旁", "杨"): "扬",
    ("提手旁", "扬"): "扬",
    ("提手旁", "羊"): "扬",
    ("提手旁", "阳"): "扬",
    ("木字旁", "扬"): "杨",
    ("木字旁", "杨"): "杨",
    ("木字旁", "羊"): "杨",
    ("木字旁", "阳"): "杨",
    ("三点水", "请"): "清",
    ("三点水", "清"): "清",
    ("言字旁", "清"): "请",
    ("言字旁", "请"): "请",
    ("言字旁", "说"): "说",
    ("女字旁", "他"): "她",
    ("女字旁", "她"): "她",
    ("单人旁", "她"): "他",
    ("单人旁", "他"): "他",
    ("单人旁", "做"): "作",
    ("单人旁", "作"): "作",
    ("立人旁", "做"): "作",
    ("立人旁", "作"): "作",
    ("宝盖头", "他"): "它",
    ("宝盖头", "它"): "它",
    ("提土旁", "的"): "地",
    ("提土旁", "地"): "地",
    ("土字旁", "的"): "地",
    ("土字旁", "地"): "地",
    ("土字底", "再"): "在",
    ("土字底", "在"): "在",
    ("日字旁", "情"): "晴",
    ("日字旁", "晴"): "晴",
    ("青字底", "晴"): "情",
    ("青字底", "情"): "情",
    ("口字旁", "问"): "吗",
    ("口字旁", "吗"): "吗",
    ("门字框", "吗"): "问",
    ("门字框", "问"): "问",
    ("双人旁", "恨"): "很",
    ("双人旁", "很"): "很",
    ("心字底", "很"): "恨",
    ("心字底", "恨"): "恨",
    ("绞丝旁", "玩"): "红",
    ("绞丝旁", "红"): "红",
    ("王字旁", "红"): "玩",
    ("王字旁", "玩"): "玩",
    ("走之底", "说"): "这",
    ("走之底", "这"): "这",
    ("竹字头", "地"): "等",
    ("竹字头", "等"): "等",
    ("穴字头", "功"): "空",
    ("穴字头", "空"): "空",
    ("工字旁", "空"): "功",
    ("工字旁", "功"): "功",
    ("白字旁", "的"): "的",
}


def _is_radical_desc(desc: str) -> bool:
    d = (desc or "").strip()
    d = _RADICAL_ALIASES.get(d, d)
    return bool(d and _RADICAL_DESC_RE.search(d))


def _canon_radical(desc: str) -> str:
    d = (desc or "").strip()
    return _RADICAL_ALIASES.get(d, d)


def _lookup_radical_glyph(desc: str, spoken: str) -> str | None:
    """Resolve glyph from radical desc + ASR surface (with aliases)."""
    rad = _canon_radical(desc)
    spoken = (spoken or "").strip()
    if not rad or len(spoken) != 1:
        return None
    hit = _RADICAL_CHAR_FIX.get((rad, spoken))
    if hit:
        return hit
    # surface alias table (羊→扬 under 提手旁)
    alias_map = _RADICAL_SURFACE_ALIAS.get(rad)
    if alias_map and spoken in alias_map:
        return alias_map[spoken]
    # also try aliased radical key in the main fix table
    if rad != desc:
        hit = _RADICAL_CHAR_FIX.get((desc, spoken))
        if hit:
            return hit
    return None


def _resolve_clarify_target(left: str, desc: str, right: str) -> str | None:
    """Resolve intended glyph from an inline clarification clause."""
    desc = (desc or "").strip()
    left = (left or "").strip()
    right = (right or "").strip()
    if not desc or len(left) != 1 or len(right) != 1:
        return None
    if _is_radical_desc(desc):
        for spoken in (right, left):
            hit = _lookup_radical_glyph(desc, spoken)
            if hit:
                return hit
        return None
    # Word disambiguation: 「名是名字的名」「己是自己的己」
    for c in (right, left):
        if c in desc:
            return c
    return right


def _strip_asr_tail_noise(text: str) -> str:
    """Drop trailing emoji / symbols ASR sometimes appends after the sentence."""
    t = (text or "").strip()
    # peel trailing junk, then restore a single sentence punct if we saw one
    end = ""
    while t:
        ch = t[-1]
        if ch in "。！？!?":
            end = ch
            t = t[:-1]
            continue
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum() or ch in "，,、：:；;":
            break
        t = t[:-1]
    return (t + end).strip()


def _parse_clarify_clause(clause: str) -> tuple[str, str, str] | None:
    c = _strip_asr_tail_noise(clause or "")
    c = re.sub(r"[。！？!?]+$", "", c).strip()
    if not c:
        return None
    m = _CLARIFY_CLAUSE_RE.fullmatch(c)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def has_name_clarify(text: str) -> bool:
    """True if ASR text looks like it contains inline glyph clarifications."""
    if not text:
        return False
    t = re.sub(r"[。！？!?]+$", "", _strip_asr_tail_noise(text)).strip()
    parts = re.split(r"[，,]", t)
    if len(parts) < 2:
        return False
    n = sum(1 for p in parts[1:] if _parse_clarify_clause(p))
    return n >= 1 and bool(_NAME_HEAD_RE.match(parts[0].strip()))


def resolve_name_clarify(text: str) -> tuple[str, str] | None:
    """Fix name glyphs from inline ASR clarifications.

    Example:
      我叫陈明阳，名是名字的名，杨是提手旁的杨。
      → 我叫陈名扬，名是名字的名，扬是提手旁的扬。
    """
    if not text:
        return None
    cleaned = _strip_asr_tail_noise(text)
    raw = cleaned.strip()
    end_punct = ""
    if raw and raw[-1] in "。！？!?":
        end_punct = raw[-1]
        body = raw[:-1]
    else:
        body = raw
    parts = [p.strip() for p in re.split(r"[，,]", body) if p.strip()]
    if len(parts) < 2:
        return None
    head = parts[0]
    hm = _NAME_HEAD_RE.match(head)
    if not hm:
        return None
    prefix, name = hm.group(1), hm.group(2)
    if not name:
        return None

    clauses: list[tuple[str, str, str, str]] = []  # raw, left, desc, right
    for p in parts[1:]:
        parsed = _parse_clarify_clause(p)
        if not parsed:
            return None  # mixed non-clarify → leave for LLM / as-is
        left, desc, right = parsed
        clauses.append((p, left, desc, right))
    if not clauses:
        return None

    targets: list[str] = []
    for _raw, left, desc, right in clauses:
        tgt = _resolve_clarify_target(left, desc, right)
        if not tgt:
            return None
        targets.append(tgt)

    n = len(targets)
    if len(name) < n:
        return None
    surname = name[:-n]
    new_name = surname + "".join(targets)

    new_clauses: list[str] = []
    for (_raw, _left, desc, _right), tgt in zip(clauses, targets):
        # show canonical radical when ASR mangled it (铁手旁→提手旁)
        disp = _canon_radical(desc) if _is_radical_desc(desc) else desc
        new_clauses.append(f"{tgt}是{disp}的{tgt}")

    out = prefix + new_name + "，" + "，".join(new_clauses) + end_punct
    if out == text.strip():
        return None
    return out, "rule_name_clarify"


# --- list / enumeration normalize → 1. 2. 3. ---
_CN_ORD_MAP = {
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
}
_LIST_NORM_INSTR = (
    "规范化",
    "加上序号",
    "列成清单",
    "整理成列表",
    "加上有序",
    "排个序",
    "加序号",
)
_YISHI_SPLIT_RE = re.compile(
    r"(?:^|[，,、；;。\s])([一二三四五六七八九十两\d]+)是"
)
_DI_ORD_SPLIT_RE = re.compile(
    r"(?:^|[，,、；;。\s])第([一二三四五六七八九十两\d]+)"
)
_INTRO_POINTS_RE = re.compile(
    r"^(.+?有[一二三四五六七八九十两\d]+[点条项个])[，,：:\s]*(.+)$"
)
_DISH_LEAD_RE = re.compile(
    r"^(.*?(?:报菜名|点(?:了|菜|个)?|菜单(?:是|如下)?|今天点了?))(.*)$"
)
_SEMI_NUM_RE = re.compile(
    r"(?:^|[，,；;\s])(\d+)[\.、．]\s*"
)


def is_list_normalize_instruction(instruction: str) -> bool:
    """True when spoken edit asks to number / normalize a list."""
    raw = (instruction or "").strip()
    if not raw:
        return False
    n = norm_text(raw)
    return any(k in raw or k in n for k in _LIST_NORM_INSTR)


# --- spoken → written (CS2W-style); fillers auto; full normalize via LLM ---
_SPOKEN2WRITTEN_INSTR = (
    "书面化",
    "改成书面语",
    "改成书面",
    "整理通顺",
    "去掉口语",
    "改正式一点",
    "说正式一点",
    "口语转书面",
    "改成正式",
    "语句规范化",
)


def is_spoken2written_instruction(instruction: str) -> bool:
    """True when user asks for spoken→written / formalization (LLM path)."""
    raw = (instruction or "").strip()
    if not raw:
        return False
    # list-normalize wins if both could match (e.g. bare 规范化)
    if is_list_normalize_instruction(raw) and not any(
        k in raw for k in ("书面", "口语", "正式", "通顺")
    ):
        return False
    n = norm_text(raw)
    return any(k in raw or k in n for k in _SPOKEN2WRITTEN_INSTR)


# Standalone pause fillers only — never fold intentional repeats (好好好好).
_FILLER_PAUSE = ("嗯", "呃", "哎", "欸", "唔", "哦")
_FILLER_PHRASE = ("那个", "就是说", "就是")
# leading: 嗯，… / 嗯 …
_FILLER_LEAD_RE = re.compile(
    r"^(?:(?:嗯|呃|哎|欸|唔|哦|那个|就是说|就是)[，,、\s]+)+"
)
# mid: …，嗯，… / …，那个，…
_FILLER_MID_RE = re.compile(
    r"[，,、]\s*(?:嗯|呃|哎|欸|唔|哦|那个|就是说|就是)(?=[，,、\s])"
)


def strip_fillers(text: str) -> tuple[str, str] | None:
    """Strip high-confidence pause fillers. Does NOT collapse intentional repeats.

    Returns (fixed, via) or None if unchanged.
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    out = raw
    while True:
        m = _FILLER_LEAD_RE.match(out)
        if not m:
            break
        out = out[m.end() :]
    prev = None
    while prev != out:
        prev = out
        out = _FILLER_MID_RE.sub("，", out)
    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"^[，,\s]+", "", out)
    out = re.sub(r"[，,、]+\s*([。！？!?]?)\s*$", r"\1", out)
    out = out.strip()
    if not out or out == raw:
        return None
    return out, "rule_filler"


def _ord_token_to_int(tok: str) -> int | None:
    tok = (tok or "").strip()
    if not tok:
        return None
    if tok.isdigit():
        return int(tok)
    if tok in _CN_ORD_MAP:
        return _CN_ORD_MAP[tok]
    try:
        return _parse_cn(tok)
    except ValueError:
        return None


def _clean_list_item(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"^[、，,；;\s]+", "", t)
    t = re.sub(r"[。！？!?，,、；;\s]+$", "", t)
    t = re.sub(r"^(?:是|、)", "", t).strip()
    return t


def _format_numbered_list(intro: str | None, items: list[str]) -> str | None:
    cleaned = [_clean_list_item(x) for x in items]
    cleaned = [x for x in cleaned if x]
    if len(cleaned) < 2:
        return None
    body = "\n".join(f"{i}. {it}" for i, it in enumerate(cleaned, 1))
    if intro:
        intro = intro.strip().rstrip("：:，,。！？!?")
        if intro:
            return f"{intro}：\n{body}"
    return body


def _split_by_markers(
    text: str, pattern: re.Pattern[str]
) -> tuple[str | None, list[str]] | None:
    """Split text by ordinal markers; return (optional_prefix, items)."""
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return None
    # markers should be increasing / unique enough
    ords: list[int] = []
    for m in matches:
        n = _ord_token_to_int(m.group(1))
        if n is None:
            return None
        ords.append(n)
    if sorted(ords) != ords or len(set(ords)) != len(ords):
        # still allow if sequential from any start
        if ords != list(range(ords[0], ords[0] + len(ords))):
            return None
    prefix = text[: matches[0].start()].strip().rstrip("，,：:")
    items: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        items.append(text[start:end])
    return (prefix or None), items


def _try_yishi_list(text: str) -> tuple[str, str] | None:
    body = re.sub(r"[。！？!?]+$", "", text.strip()).strip()
    intro = None
    rest = body
    m = _INTRO_POINTS_RE.match(body)
    if m:
        intro, rest = m.group(1).strip(), m.group(2).strip()
    split = _split_by_markers(rest, _YISHI_SPLIT_RE)
    if not split:
        # whole-body 一是… without intro peel
        split = _split_by_markers(body, _YISHI_SPLIT_RE)
        if not split:
            return None
        prefix, items = split
        intro = intro or prefix
    else:
        prefix, items = split
        if prefix and not intro:
            intro = prefix
    out = _format_numbered_list(intro, items)
    if not out or out == text.strip():
        return None
    return out, "rule_list_normalize"


def _try_di_ord_list(text: str) -> tuple[str, str] | None:
    body = re.sub(r"[。！？!?]+$", "", text.strip()).strip()
    split = _split_by_markers(body, _DI_ORD_SPLIT_RE)
    if not split:
        return None
    prefix, items = split
    # strip leading 「是」 after 第一
    items = [re.sub(r"^是", "", it).strip() for it in items]
    out = _format_numbered_list(prefix, items)
    if not out or out == text.strip():
        return None
    return out, "rule_list_normalize"


def _try_dunhao_list(text: str) -> tuple[str, str] | None:
    body = re.sub(r"[。！？!?]+$", "", text.strip()).strip()
    if "、" not in body:
        return None
    parts = body.split("、")
    if len(parts) < 2:
        return None
    first = parts[0].strip()
    intro = None
    item0 = first
    m = _DISH_LEAD_RE.match(first)
    if m and m.group(2).strip():
        # keep lead only as cue; dish menus usually omit the lead in output
        item0 = m.group(2).strip()
        intro = None
    elif re.search(r"(报菜名|点(?:了|菜|个)|菜单)", first):
        # lead only, first dish starts after punctuation
        return None
    else:
        # plain顿号 list without dish lead: only if short items (菜名风格)
        if any(len(p) > 12 for p in parts):
            return None
        intro = None
        item0 = first
    items = [item0] + [p.strip() for p in parts[1:]]
    out = _format_numbered_list(intro, items)
    if not out or out == text.strip():
        return None
    return out, "rule_list_normalize"


def _try_unify_semi_numbered(text: str) -> tuple[str, str] | None:
    """Unify inline/newline 1、/1. enumerations into newline `1. ` form."""
    body = re.sub(r"[。！？!?]+$", "", text.strip()).strip()
    if "\n" in body:
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        numbered = [ln for ln in lines if re.match(r"^\d+[\.、．]\s*\S", ln)]
        if len(numbered) >= 2:
            intro_lines = [ln for ln in lines if not re.match(r"^\d+[\.、．]", ln)]
            items = [re.sub(r"^\d+[\.、．]\s*", "", ln) for ln in numbered]
            out = _format_numbered_list(
                intro_lines[0] if intro_lines else None, items
            )
            if out and out != text.strip():
                return out, "rule_list_normalize"
            return None
    matches = list(_SEMI_NUM_RE.finditer(body))
    if len(matches) < 2:
        return None
    prefix = body[: matches[0].start()].strip().rstrip("，,：:")
    items: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        items.append(body[start:end])
    out = _format_numbered_list(prefix or None, items)
    if not out or out == text.strip():
        return None
    return out, "rule_list_normalize"


def normalize_list_structure(text: str) -> tuple[str, str] | None:
    """Normalize spoken enumerations into newline `1. 2. 3.` lists.

    Examples:
      下面有三点，一是开会，二是吃饭，三是睡觉
      → 下面有三点：\\n1. 开会\\n2. 吃饭\\n3. 睡觉
      今天点宫保鸡丁、鱼香肉丝、麻婆豆腐
      → 1. 宫保鸡丁\\n2. 鱼香肉丝\\n3. 麻婆豆腐
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    for fn in (
        _try_yishi_list,
        _try_di_ord_list,
        _try_dunhao_list,
        _try_unify_semi_numbered,
    ):
        got = fn(raw)
        if got:
            return got
    return None


def find_near_span(draft: str, needle: str) -> tuple[int, int] | None:
    """Same-length (or ±1) window in draft close to needle but not equal."""
    needle = needle or ""
    n = len(needle)
    if n < 2 or not draft:
        return None
    max_ed = max(1, n // 3)
    best: tuple[int, int, int, int] | None = None  # dist, len_delta, start, end
    lengths = [n, n + 1]
    if n > 2:
        lengths.append(n - 1)
    for length in lengths:
        if length < 2 or length > len(draft):
            continue
        for i in range(0, len(draft) - length + 1):
            window = draft[i : i + length]
            dist = edit_distance(window, needle)
            if dist < 1 or dist > max_ed:
                continue
            cand = (dist, abs(length - n), i, i + length)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None
    return best[2], best[3]


def apply_fuzzy_replace(draft: str, instruction: str) -> str | None:
    """Replace a near-match span when ASR wrote the right glyph or a near src."""
    if not draft or not instruction:
        return None
    pairs = extract_replace_pairs(instruction, keep_equal=True)
    if not pairs:
        return None
    out = draft
    hits = 0
    for src, dst in sorted(pairs, key=lambda x: -len(x[0])):
        if src == dst:
            # Correct glyph already in the draft: nothing to repair.
            if src in out:
                continue
            span = find_near_span(out, src)
            repl = src
        elif src not in out:
            span = find_near_span(out, src)
            repl = dst
        else:
            continue
        if span is None:
            continue
        i, j = span
        nxt = out[:i] + repl + out[j:]
        if nxt != out:
            out = nxt
            hits += 1
    return out if hits else None


def _split_sentences(t: str) -> list[str]:
    """Split keeping trailing sentence punct on each piece."""
    if not t:
        return []
    parts = re.findall(r"[^。！？!?]+[。！？!?]?|[。！？!?]", t)
    return [p for p in parts if p and p.strip()]


def _collapse_sentence_loops(t: str, draft: str = "") -> str:
    sents = _split_sentences(t)
    if len(sents) < 2:
        return t

    # adjacent identical sentences → keep up to first repeat
    out: list[str] = []
    for s in sents:
        if out and norm_text(s) == norm_text(out[-1]) and norm_text(s):
            break
        out.append(s)
    if len(out) < len(sents):
        return "".join(out).strip()

    # ABAB… (≥2 full periods) → prefer A if closer to draft else shorter
    if len(sents) >= 4:
        a, b = sents[0], sents[1]
        na, nb = norm_text(a), norm_text(b)
        if na and nb and na != nb:
            periods = 0
            ok = True
            i = 0
            while i + 1 < len(sents):
                if norm_text(sents[i]) == na and norm_text(sents[i + 1]) == nb:
                    periods += 1
                    i += 2
                else:
                    ok = False
                    break
            if ok and periods >= 2 and (i == len(sents) or i == len(sents) - 1):
                # trailing single A also ok
                if i == len(sents) - 1 and norm_text(sents[i]) != na:
                    pass
                else:
                    draft_n = norm_text(draft)
                    if draft_n and na == draft_n:
                        return b.strip() if nb != draft_n else a.strip()
                    if draft_n and nb == draft_n:
                        return a.strip()
                    # prefer the one that looks like an edit result vs draft flip
                    return (a if len(norm_text(a)) <= len(norm_text(b)) else b).strip()
    return "".join(out).strip() if out else t


def _strip_prompt_leak(t: str) -> str:
    marker = "【改写结果】"
    if marker in t:
        after = t.rsplit(marker, 1)[-1].strip()
        if after:
            t = after
        else:
            t = t[: t.rfind(marker)].strip()
    for leak in ("【草稿】", "【修改指令】"):
        if leak in t:
            t = t.split("【", 1)[0].strip()
            break
    t = re.sub(r"【[^】]*】", "", t).strip()
    return t


def clean_rewrite(text: str, *, draft: str = "") -> str:
    """Strip special tokens / prompt leak / repetition loops from rewrite output."""
    t = (text or "").strip()
    t = re.sub(r"<\|[^|>]*\|>?", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _strip_prompt_leak(t)
    for prefix in ("改写结果：", "改写结果:", "结果：", "结果:", "输出：", "输出:"):
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()
    if not t:
        return ""

    t = _collapse_sentence_loops(t, draft=draft)

    # whole-string prefix repeats (no sentence punct)
    for length in range(1, min(48, len(t) // 2) + 1):
        unit = t[:length]
        if len(unit) < 1:
            continue
        repeats = 0
        pos = 0
        while t.startswith(unit, pos):
            repeats += 1
            pos += length
        if repeats >= 2 and pos >= len(t) * 0.8:
            t = unit.strip()
            break

    limit = max(48, len(draft) * 3) if draft else 48
    if len(t) > limit:
        m = re.match(r"^[\u4e00-\u9fffA-Za-z0-9，。、：:（）()\-]+", t)
        if m:
            cand = m.group(0).rstrip("，。、：:")
            if cand and cand != draft:
                t = cand[:limit]
            else:
                t = t[:limit].strip()
        else:
            t = t[:limit].strip()

    t = re.sub(r"[。！？!?]+$", "", t).strip()
    return t


def looks_like_garbage(text: str, draft: str) -> bool:
    if not text:
        return True
    if "<|" in text or "|>" in text:
        return True
    if "【" in text or "】" in text:
        return True
    sents = _split_sentences(text)
    if len(sents) >= 3:
        norms = [norm_text(s) for s in sents if norm_text(s)]
        if norms and max(norms.count(n) for n in set(norms)) >= 3:
            return True
    if draft and len(text) > max(80, len(draft) * 5):
        return True
    return False


def clean_spoken2written(text: str, *, draft: str = "") -> str:
    """Clean LLM spoken→written output; collapse multi-sentence restarts to one best line."""
    t = (text or "").strip()
    if not t:
        return ""
    # strip spoken2written prompt leaks
    for marker in ("【书面语】", "【原文】", "书面语：", "书面语:", "原文：", "原文:"):
        if marker in t:
            if marker.startswith("【"):
                after = t.rsplit(marker, 1)[-1].strip() if marker in t else t
                # keep text before first 【 if model echoed both
                before = t.split("【", 1)[0].strip()
                t = after or before or t
            elif t.startswith(marker):
                t = t[len(marker) :].strip()
    t = clean_rewrite(t, draft=draft)
    if not t:
        return ""

    draft_n = norm_text(draft)
    sents = [s.strip() for s in _split_sentences(t) if norm_text(s)]
    # Model often restarts 2–3 near-paraphrases in one string → pick closest to draft
    if len(sents) >= 2 or (draft_n and len(norm_text(t)) > int(len(draft_n) * 1.35) + 2):
        if not sents:
            return t
        best = sents[0]
        best_key: tuple[float, int, int] | None = None
        for s in sents:
            sn = norm_text(s)
            if not sn:
                continue
            # prefer moderate edit distance to draft (rewrite, not echo of instruction junk)
            dist = edit_distance(draft_n, sn) / max(len(draft_n), 1) if draft_n else 0.0
            # penalize dropping intensifier 千万 when draft had it
            penalty = 0.0
            if draft and "千万" in draft and "千万" not in s:
                penalty += 0.35
            if draft and "不要" in draft and "不要" not in s and "别" not in s:
                penalty += 0.25
            # prefer not much longer than draft
            len_pen = max(0, len(sn) - len(draft_n)) if draft_n else 0
            key = (dist + penalty, len_pen, len(sn))
            if best_key is None or key < best_key:
                best_key = key
                best = s
        t = re.sub(r"[。！？!?]+$", "", best).strip()
    return t


def looks_like_bad_spoken2written(text: str, draft: str) -> bool:
    """True if cleaned written still looks like a failed / looped rewrite."""
    if looks_like_garbage(text, draft):
        return True
    if not text or not draft:
        return not text
    tn, dn = norm_text(text), norm_text(draft)
    if not tn:
        return True
    # still multiple sentences after clean → bad
    if len(_split_sentences(text)) >= 2:
        return True
    if len(tn) > len(dn) * 1.6 + 4:
        return True
    # drifted too far from draft
    if edit_distance(dn, tn) / max(len(dn), 1) > 0.55:
        return True
    return False


def is_echo_fail(pred: str, instruction: str) -> bool:
    p = norm_text(pred)
    i = norm_text(instruction)
    if not p or not i:
        return False
    if p == i:
        return True
    if len(i) >= 4 and i in p and len(p) <= len(i) + 2:
        return True
    return False


# Idle / chat / filler instructions → keep draft unchanged.
_KEEP_EXACT = frozenset(
    {
        "嗯",
        "嗯嗯",
        "啊",
        "哦",
        "好的",
        "收到",
        "好的收到",
        "好的好的",
        "哈哈",
        "哈哈哈",
        "谢谢",
        "谢谢你",
        "你是谁",
        "一二三四五",
    }
)
_KEEP_SUBSTR = (
    "今天天气",
    "讲个笑话",
    "无关指令",
    "给我讲个",
)


def should_keep_draft(draft: str, instruction: str) -> bool:
    """True when the spoken edit should leave the draft untouched (noop)."""
    instr_n = norm_text(instruction)
    if not instr_n:
        return True
    if instr_n in _KEEP_EXACT:
        return True
    if len(instr_n) <= 2 and re.fullmatch(r"[嗯啊哦唔欸哎哈]+", instr_n):
        return True
    for sub in _KEEP_SUBSTR:
        if sub in instr_n:
            return True
    if apply_radical_describe_replace(draft, instruction):
        return False
    # 「把A改成B」but A not in draft → keep, unless a near span in the draft matches.
    pairs = extract_replace_pairs(instruction)
    if pairs and not any(src in draft for src, _dst in pairs):
        if apply_fuzzy_replace(draft, instruction):
            return False
        return True
    return False


# Bidirectional polarity / switch pairs (longer phrases first when matching).
_TOGGLE_PAIRS: list[tuple[str, str]] = [
    ("打开自动亮度", "关闭自动亮度"),
    ("打开静音", "关闭静音"),
    ("打开热点", "关闭热点"),
    ("打开定位", "关闭定位"),
    ("打开蓝牙", "关闭蓝牙"),
    ("继续播放", "暂停播放"),
    ("开始录音", "停止录音"),
    ("解锁", "锁屏"),
    ("打开", "关闭"),
    ("打开", "关掉"),
    ("打开", "关上"),
    ("继续", "暂停"),
    ("开始", "停止"),
]


def _toggle_antonym_map() -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for a, b in _TOGGLE_PAIRS:
        m.setdefault(a, [])
        m.setdefault(b, [])
        if b not in m[a]:
            m[a].append(b)
        if a not in m[b]:
            m[b].append(a)
    for k in m:
        m[k] = sorted(set(m[k]), key=len, reverse=True)
    return m


_TOGGLE_ANTONYMS = _toggle_antonym_map()
_TOGGLE_PHRASES = sorted(_TOGGLE_ANTONYMS.keys(), key=len, reverse=True)


def apply_toggle(draft: str, instruction: str) -> str | None:
    """Flip polarity words in draft per spoken toggle instruction (改成打开/关闭…)."""
    if not draft or not instruction:
        return None
    instr = (instruction or "").strip()
    instr = re.sub(r"[。！？!?]+$", "", instr).strip()
    instr = re.sub(r"^(请)?(改成|变成|换成)", "", instr).strip()
    if not instr:
        return None
    instr_n = norm_text(instr)

    # Prefer longer phrase matches in the instruction; fall back to shorter polarity words.
    targets = [p for p in _TOGGLE_PHRASES if p in instr or p in instr_n]
    for target in targets:
        for opp in _TOGGLE_ANTONYMS.get(target, []):
            if opp in draft:
                out = draft.replace(opp, target, 1)
                if out != draft:
                    return out
    return None


def apply_edit_rules(draft: str, instruction: str) -> tuple[str, str] | None:
    """Rule cascade: list-norm cmd, noop, toggle, radical, exact, fuzzy. None → LLM."""
    # 「规范化 / 加上序号」→ rewrite draft structure (rules only, no LLM)
    if is_list_normalize_instruction(instruction):
        got = normalize_list_structure(draft)
        if got:
            return got
        return draft, "noop"
    if should_keep_draft(draft, instruction):
        return draft, "noop"
    for fn, via in (
        (apply_toggle, "rule_toggle"),
        (apply_radical_describe_replace, "rule_radical"),
        (apply_simple_replace, "rule"),
        (apply_fuzzy_replace, "rule_fuzzy"),
    ):
        out = fn(draft, instruction)
        if out and out != draft:
            return out, via
    return None


def _self_check() -> None:
    assert clean_rewrite("解锁。【改写结果】") == "解锁"
    assert clean_rewrite("导航去浦东机场。【改写结果】") == "导航去浦东机场"
    assert (
        clean_rewrite("继续播放。【草稿】暂停播放 【修改指令】改成继续播放。 【改写结果】")
        == "继续播放"
    )
    assert clean_rewrite("打开支付宝。打开支付宝。打开支付宝。") == "打开支付宝"
    out = clean_rewrite(
        "关闭热点。打开热点。关闭热点。打开热点。关闭热点。",
        draft="打开热点",
    )
    assert out in ("关闭热点", "打开热点"), out
    assert norm_text("请把灯打开。") == norm_text("请把灯打开")
    assert not looks_like_garbage("打开服务器", "打开窗帘")
    assert looks_like_garbage("解锁。【改写结果】", "锁屏")

    # multi-clause replace
    assert extract_replace_pairs("把灯改成空调，把窗帘改成窗户") == [
        ("灯", "空调"),
        ("窗帘", "窗户"),
    ]
    multi_cases = [
        ("打开客厅的灯和窗帘", "把灯改成空调，把窗帘改成窗户", "打开客厅的空调和窗户"),
        ("明天去上海见小明", "把上海改成北京，把小明改成小红", "明天去北京见小红"),
        ("播放周杰伦的七里香", "把周杰伦改成邓紫棋，把七里香改成光年之外", "播放邓紫棋的光年之外"),
        ("设置闹钟七点半提醒开会", "把七点半改成八点，把开会改成吃药", "设置闹钟八点提醒吃药"),
        ("导航去公司然后打开微信", "把公司改成机场，把微信改成电话", "导航去机场然后打开电话"),
        ("查询明天北京天气", "把明天改成后天，把北京改成上海", "查询后天上海天气"),
        ("打开卧室的灯和空调", "把卧室改成客厅，把空调改成窗帘", "打开客厅的灯和窗帘"),
        ("发消息给张三说我到了", "把张三改成李四，把我到了改成晚点到", "发消息给李四说晚点到"),
        ("预约明天下午三点会议室", "把明天下午改成后天上午，把三点改成十点", "预约后天上午十点会议室"),
        ("关闭所有灯和音乐", "把灯改成窗户，把音乐改成空调", "关闭所有窗户和空调"),
    ]
    for draft, instr, gold in multi_cases:
        got = apply_simple_replace(draft, instr)
        assert got == gold, (draft, instr, got, gold)

    # single replace regression
    assert apply_simple_replace("打开窗帘", "把窗帘变成服务器") == "打开服务器"
    assert apply_simple_replace("请把灯关掉", "改成打开") is None  # no pair / no src
    assert apply_simple_replace("查询护身三百", "把护身改成护身") is None  # A→A skipped
    # ASR pause comma before 改成
    assert (
        apply_simple_replace(
            "发消息给张三说我到了",
            "把张三改成李四，把我到了，改成晚点到。",
        )
        == "发消息给李四说晚点到"
    )

    # noop keep
    noop_cases = [
        ("打开窗帘", "今天天气怎么样"),
        ("导航去公司", "给我讲个笑话"),
        ("播放音乐", "你是谁"),
        ("设置闹钟七点", "一二三四五"),
        ("查询天气", "谢谢你"),
        ("打开微信", "嗯"),
        ("关闭空调", "好的收到"),
        ("提醒我开会", "哈哈哈"),
        ("拨打一二零", "无关指令测试"),
        ("发送消息", "把香蕉变成苹果"),
    ]
    for draft, instr in noop_cases:
        assert should_keep_draft(draft, instr), (draft, instr)
    assert not should_keep_draft("打开窗帘", "把窗帘变成服务器")
    assert not should_keep_draft("打开客厅的灯和窗帘", "把灯改成空调，把窗帘改成窗户")

    # toggle polarity
    toggle_cases = [
        ("请把灯关掉", "改成打开", "请把灯打开"),
        ("打开蓝牙", "改成关闭", "关闭蓝牙"),
        ("关闭静音", "改成打开静音", "打开静音"),
        ("开始录音", "改成停止录音", "停止录音"),
        ("打开定位", "改成关闭定位", "关闭定位"),
        ("关闭自动亮度", "改成打开自动亮度", "打开自动亮度"),
        ("暂停播放", "改成继续播放", "继续播放"),
        ("打开热点", "改成关闭热点", "关闭热点"),
        ("锁屏", "改成解锁", "解锁"),
        ("静音模式打开", "改成关闭静音", "静音模式关闭"),
    ]
    for draft, instr, gold in toggle_cases:
        got = apply_toggle(draft, instr)
        assert got == gold, (draft, instr, got, gold)
    assert apply_toggle("打开窗帘", "把窗帘变成服务器") is None

    glyph_cases = [
        ("到的了", "把白字旁的的修改为提土旁的地", "到地了"),
        ("请你过来", "把言字旁的请改成三点水的清", "清你过来"),
        ("她去了", "把女字旁的她改成单人旁的他", "他去了"),
        ("他是猫", "把单人旁的他改成宝盖头的它", "它是猫"),
        ("做完了", "把单人旁的做改成立人旁的作", "作完了"),
        ("杨树", "把木字旁的杨改成提手旁的扬", "扬树"),
        ("晴天", "把日字旁的晴改成青字底的情", "情天"),
        ("问他", "把门字框的问改成口字旁的吗", "吗他"),
        ("很忙", "把双人旁的很改成心字底的恨", "恨忙"),
        ("红了", "把绞丝旁的红改成王字旁的玩", "玩了"),
        ("这是", "把走之底的这改成言字旁的说", "说是"),
        ("等一下", "把竹字头的等改成土字旁的地", "地一下"),
        ("空的", "把穴字头的空改成工字旁的功", "功的"),
        ("到的了", "把的改成提土旁的地", "到地了"),
        ("再公司见", "把再改成土字底的在", "在公司见"),
        ("做业", "把做改成单人旁的作", "作业"),
        ("已知道", "把已改成自己的己", "己知道"),
        ("到的了", "把的改成土也地", "到地了"),
        ("做完", "把做改成单人作", "作完"),
    ]
    for draft, instr, gold in glyph_cases:
        got = apply_radical_describe_replace(draft, instr)
        assert got == gold, (draft, instr, got, gold)
    assert apply_radical_describe_replace("她走了", "把女字旁的她改成他") is None
    assert apply_radical_describe_replace("清问", "把三点水的清改成请") is None

    assert apply_fuzzy_replace("播放周杰轮的歌", "把周杰伦改成周杰伦") == "播放周杰伦的歌"
    assert apply_fuzzy_replace("导航到公词", "把宫词改成公司") == "导航到公司"
    assert apply_fuzzy_replace("查询护身三百", "把护身改成护身") is None
    assert not should_keep_draft("导航到公词", "把宫词改成公司")
    assert should_keep_draft("发送消息", "把香蕉变成苹果")

    assert normalize_display("一二零") == "120"
    assert normalize_display("二十七") == "27"
    assert normalize_display("三百") == "300"
    assert normalize_display("两点五") == "2.5"
    assert normalize_display("三点十分") == "3:10"
    assert normalize_display("七点") == "7点"
    assert normalize_display("百分之二十") == "20%"
    assert normalize_display("一加一等于二") == "1 + 1 = 2"
    assert normalize_display("二乘三等于六") == "2 × 3 = 6"
    assert normalize_display("十减三等于七") == "10 - 3 = 7"
    assert normalize_display("八除以二等于四") == "8 ÷ 2 = 4"
    assert normalize_display("一加一") == "1 + 1"
    assert normalize_display("三点水") == "三点水"
    assert normalize_display("井号和艾特") == "#和@"
    assert normalize_display("1 + 1 = 2") == "1 + 1 = 2"
    assert normalize_display("拨打一二零") == "拨打120"
    # 千万 = intensifier, not 10000000
    assert normalize_display("千万不要不要有贪婪的意识") == "千万不要不要有贪婪的意识"
    assert normalize_display("千万别去") == "千万别去"
    assert normalize_display("一万") == "10000"
    assert normalize_display("赚了千万") == "赚了10000000"  # bare 千万 after verb ≈ number

    name_cases = [
        (
            "我叫陈明阳，名是名字的名，杨是提手旁的杨。",
            "我叫陈名扬，名是名字的名，扬是提手旁的扬。",
        ),
        (
            "我叫陈明阳，名是名字的名，杨是提手旁的杨",
            "我叫陈名扬，名是名字的名，扬是提手旁的扬",
        ),
        (
            "我是李小明，明是明日的明",
            "我是李小明，明是明日的明",  # already correct → None
        ),
    ]
    got = resolve_name_clarify(name_cases[0][0])
    assert got is not None and got[0] == name_cases[0][1] and got[1] == "rule_name_clarify", got
    got = resolve_name_clarify(name_cases[1][0])
    assert got is not None and got[0] == name_cases[1][1], got
    assert resolve_name_clarify(name_cases[2][0]) is None  # unchanged
    # trailing emoji from SenseVoice must not block the rule path
    got = resolve_name_clarify("我叫陈明阳，名是名字的名，杨是提手旁的杨。😊")
    assert got is not None and got[0] == name_cases[0][1], got
    assert has_name_clarify(name_cases[0][0])
    assert not has_name_clarify("打开窗帘")
    assert resolve_name_clarify("打开窗帘") is None
    # ASR mangled radical/surface: 提手旁的扬 → 铁手旁的羊
    got = resolve_name_clarify("我是陈名杨，名是名字的名，杨是铁手旁的羊")
    assert got is not None, got
    assert got[0] == "我是陈名扬，名是名字的名，扬是提手旁的扬", got

    list_gold = "下面有三点：\n1. 开会\n2. 吃饭\n3. 睡觉"
    got = normalize_list_structure("下面有三点，一是开会，二是吃饭，三是睡觉")
    assert got is not None and got[0] == list_gold and got[1] == "rule_list_normalize", got
    got = normalize_list_structure("今天点宫保鸡丁、鱼香肉丝、麻婆豆腐")
    assert got is not None and got[0] == "1. 宫保鸡丁\n2. 鱼香肉丝\n3. 麻婆豆腐", got
    got = normalize_list_structure("第一开会，第二吃饭，第三睡觉")
    assert got is not None and "1. 开会" in got[0] and "3. 睡觉" in got[0], got
    assert normalize_list_structure("打开窗帘") is None
    assert is_list_normalize_instruction("规范化一下")
    assert is_list_normalize_instruction("加上序号")
    ruled = apply_edit_rules(
        "下面有三点，一是开会，二是吃饭，三是睡觉", "规范化一下"
    )
    assert ruled is not None and ruled[0] == list_gold, ruled

    # fillers: strip pause words; never fold intentional repeats
    got = strip_fillers("嗯，打开窗帘")
    assert got is not None and got[0] == "打开窗帘" and got[1] == "rule_filler", got
    got = strip_fillers("呃，那个，今天开会")
    assert got is not None and got[0] == "今天开会", got
    assert strip_fillers("打开窗帘") is None
    assert strip_fillers("好好好好") is None
    assert strip_fillers("加油加油") is None
    assert strip_fillers("打开打开窗帘") is None  # intentional/ambiguous → keep

    assert is_spoken2written_instruction("书面化一下")
    assert is_spoken2written_instruction("改成书面语")
    assert is_spoken2written_instruction("整理通顺")
    assert not is_spoken2written_instruction("把窗帘改成空调")
    assert not is_spoken2written_instruction("规范化一下")  # list normalize

    # LLM often emits 2–3 near-duplicate rewrites; keep best single line
    looped = "千万不要有贪婪的意识。不要不要有贪婪的意识。 不要有贪婪的意识"
    draft_s2w = "千万不要不要有贪婪的意识。"
    got = clean_spoken2written(looped, draft=draft_s2w)
    assert got == "千万不要有贪婪的意识", got
    assert not looks_like_bad_spoken2written(got, draft_s2w)

    print("voice_edit_lib self_check ok")


if __name__ == "__main__":
    _self_check()
