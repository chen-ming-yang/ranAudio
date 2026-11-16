#!/usr/bin/env python3
"""Synthetic SFT data: ASR text with 组词/偏旁 clarifications → corrected text.

Train must not contain the name 陈名扬. Each train sentence may clarify one
character or two. The two-character rows include a keep-vs-edit contrast
(for example 明是明日的明 stays, while the other character changes).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from pypinyin import lazy_pinyin

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from voice_edit_lib import _RADICAL_ALIASES, _RADICAL_CHAR_FIX  # noqa: E402

KIMI = Path(__file__).resolve().parents[1]
OUT_DIR = KIMI / "eval_name_clarify" / "sft"

INSTRUCTION = (
    "下面是语音识别原文。若句中用组词或偏旁说明字形，"
    "把被说明的字和说明本身改成同一个正确字。"
    "若用「第N条」「第N点」或「下一条」逐条列举，收成从 1. 开始、每条一行的编号。"
    "若在报菜名或罗列一串名称，每个名称单独一行，不加序号。"
    "若用两个部件拼出这个字，把被说明的字改成这个字。"
    "被说明的字认成了别的字时，以说明为准。"
    "若用「第N段」或「下一段」分段，或前后说的是不同的事，分成若干段，段与段之间空一行。"
    "钟点写成「时:分」，算式写成数字和加减乘除符号，句里的数目写成阿拉伯数字。"
    "「第N条」「第N段」仍按列表和分段处理。"
    "没有这类情况则原样输出。只输出全文。\n"
    "【原文】{asr}"
)
OLD_INSTRUCTION_HEAD = (
    "下面是语音识别原文。若句中用组词或偏旁说明字形，"
    "把被说明的字和说明本身改成同一个正确字。"
    "没有这类说明则原样输出。只输出全文。\n"
    "【原文】"
)

BANNED_NAME_CHARS = set("的吗了是呢吧啊呀嘛")
SURNAMES = list("赵钱孙李周吴郑王冯褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳唐罗")
# 陈 is allowed in generated names, but the full string 陈名扬 is forbidden in train.

WORDS = [
    ("名", "名字"), ("明", "明天"), ("明", "明日"), ("己", "自己"), ("已", "已经"),
    ("和", "和平"), ("力", "力量"), ("方", "方向"), ("中", "中国"), ("时", "时间"),
    ("东", "东西"), ("乐", "快乐"), ("美", "美丽"), ("安", "安静"), ("友", "朋友"),
    ("学", "学习"), ("生", "生活"), ("希", "希望"), ("未", "未来"), ("成", "成功"),
    ("海", "海洋"), ("文", "文章"), ("语", "语言"), ("思", "思想"), ("情", "情感"),
    ("智", "智慧"), ("清", "清楚"), ("青", "青春"), ("请", "请求"), ("扬", "飞扬"),
    ("杨", "杨树"), ("阳", "阳光"), ("在", "存在"), ("再", "再次"), ("他", "他们"),
    ("她", "她们"), ("它", "它们"), ("做", "做事"), ("作", "作业"), ("红", "红色"),
    ("很", "很多"), ("恨", "仇恨"), ("玩", "玩耍"), ("问", "问题"), ("吗", "干吗"),
    ("晴", "晴天"), ("空", "天空"), ("功", "功夫"), ("等", "等待"), ("这", "这里"),
    ("西", "西方"), ("习", "学习"), ("丽", "美丽"), ("鱼", "鱼肉"), ("京", "北京"),
    ("州", "州府"), ("河", "河流"), ("街", "街道"), ("路", "道路"), ("德", "道德"),
    ("华", "中华"), ("子", "子女"), ("蒸", "蒸汽"), ("香", "香味"), ("丝", "丝绸"),
    ("丁", "丁香"), ("宫", "宫殿"), ("保", "保护"), ("鸡", "鸡肉"),
]

BANK = "".join(dict.fromkeys("".join(c + w for c, w in WORDS) + "明名扬杨阳羊青清请己已和力方"))

# (radical_as_spoken, surface, gold_char, gold_radical)
# gold_radical may differ when the spoken radical is an alias.
RADICALS: list[tuple[str, str, str, str]] = []
for (rad, surface), gold in _RADICAL_CHAR_FIX.items():
    if surface in BANNED_NAME_CHARS and gold in BANNED_NAME_CHARS:
        continue
    RADICALS.append((rad, surface, gold, rad))
for alias, canon in _RADICAL_ALIASES.items():
    for (rad, surface), gold in _RADICAL_CHAR_FIX.items():
        if rad == canon and surface != gold:
            RADICALS.append((alias, surface, gold, canon))

PLAINS = [
    "今天天气不错，适合出门散步。",
    "今天下午三点开会。",
    "把文件发到我邮箱。",
    "明天去超市买牛奶。",
    "这首歌很好听。",
    "晚上七点到家。",
    "帮我订一张明天的票。",
    "会议室在二楼。",
    "记得带伞。",
    "这个方案先这样。",
    "午饭吃面条就行。",
    "周末打算在家休息。",
    "请把灯关掉。",
    "我一会儿就到。",
    "报告放在桌面上。",
    "导航已经开了。",
    "密码我稍后再发。",
    "今天先到这里。",
    "来一份宫保鸡丁。",
    "地铁在路口左转。",
    "公司九点上班。",
    "书放回书架。",
    "水温刚刚好。",
    "比赛周六开始。",
    "发票开公司抬头。",
]

MUST_VAL = [
    ("我叫陈明阳，明是名字的明，阳是提手旁的阳。", "我叫陈名扬，名是名字的名，扬是提手旁的扬。"),
    ("我叫陈明阳，名是名字的名，杨是提手旁的杨。", "我叫陈名扬，名是名字的名，扬是提手旁的扬。"),
    ("我是陈名杨，名是名字的名，杨是铁手旁的羊", "我是陈名扬，名是名字的名，扬是提手旁的扬"),
    ("我是李小明，明是明日的明", "我是李小明，明是明日的明"),
    ("今天天气不错，适合出门散步。", "今天天气不错，适合出门散步。"),
]

LIST_MUST = [
    ("第一条买牛奶，第二条回家，下一条休息", "1. 买牛奶\n2. 回家\n3. 休息"),
    ("帮我记一下，第一点开会，下一条写报告", "帮我记一下\n1. 开会\n2. 写报告"),
    ("合同第一条不作修改。", "合同第一条不作修改。"),
    ("报个菜名：宫保鸡丁、鱼香肉丝、麻婆豆腐", "报个菜名：宫保鸡丁\n鱼香肉丝\n麻婆豆腐"),
    ("我去公司，然后回家。", "我去公司，然后回家。"),
]

PARA_MUST = [
    ("第一段今天下午开会，讨论合同。第二段晚上回家吃饭。", "今天下午开会，讨论合同。\n\n晚上回家吃饭。"),
    ("第一段今天天气不错。下一段合同已经签完，下午寄出。", "今天天气不错。\n\n合同已经签完，下午寄出。"),
    ("今天天气不错，适合出门。另外合同已经签完，下午寄出。", "今天天气不错，适合出门。\n\n合同已经签完，下午寄出。"),
    ("今天下午开会，晚上把纪要写完。", "今天下午开会，晚上把纪要写完。"),
    ("我去公司，然后回家。", "我去公司，然后回家。"),
]

NEW_MUST = [
    ("我姓程，程是耳东成", "我姓陈，陈是耳东陈"),
    ("方是提手旁的样", "扬是提手旁的扬"),
    ("我姓陈，陈是耳东陈", "我姓陈，陈是耳东陈"),
    ("今天先去公园散步。", "今天先去公园散步。"),
    ("现在五点四十", "现在5:40"),
    ("一加二等于三", "1 + 2 = 3"),
    ("这个偏旁是三点水。", "这个偏旁是三点水。"),
]

SPLITS = [
    ("陈", "耳东"), ("李", "木子"), ("张", "弓长"), ("许", "言午"), ("胡", "古月"),
    ("吴", "口天"), ("章", "立早"), ("刘", "文刀"), ("林", "双木"), ("明", "日月"),
]
RADICAL_GOLD = [
    ("提手旁", "扬"), ("木字旁", "杨"), ("三点水", "清"), ("言字旁", "请"),
    ("女字旁", "她"), ("单人旁", "他"), ("宝盖头", "它"), ("提土旁", "地"),
    ("日字旁", "晴"), ("口字旁", "吗"), ("门字框", "问"), ("双人旁", "很"),
    ("心字底", "恨"), ("绞丝旁", "红"), ("王字旁", "玩"), ("走之底", "这"),
    ("竹字头", "等"), ("穴字头", "空"), ("工字旁", "功"),
]
WRONG_CHARS = "方样成程尘冬石马牛鱼鸟花草山水火金"
_D = "零一二三四五六七八九"

MATTERS = [
    ["今天下午开会。", "会上讨论合同。", "晚上把纪要写完。"],
    ["今天天气不错。", "适合出门散步。", "出门记得带伞。"],
    ["合同已经签完。", "下午寄出去。", "回执明天到。"],
    ["晚上回家吃饭。", "先把菜热一下。", "饭后洗碗。"],
    ["明天出差。", "机票已经订好。", "后天回来。"],
    ["快递到了驿站。", "下班去取。", "取件码在短信里。"],
    ["孩子放学比较早。", "我去学校接。", "接完直接回家。"],
    ["周末要去医院。", "记得带医保卡。", "挂号已经约好。"],
    ["早上先买菜。", "中午自己做饭。", "晚上还有剩菜。"],
    ["报告周五交。", "数据还要核对。", "图表今晚补上。"],
]

_CN = "一二三四五六七八九十"
TASKS = ["买牛奶", "回家", "开会", "写报告", "发邮件", "取快递", "订机票", "交文件", "打电话", "看合同", "整理桌子", "预约医生", "休息", "买菜", "交作业"]
DISHES = ["宫保鸡丁", "鱼香肉丝", "麻婆豆腐", "糖醋排骨", "红烧肉", "清蒸鲈鱼", "西红柿鸡蛋", "酸菜鱼", "回锅肉", "水煮牛肉", "青椒肉丝", "蛋炒饭", "小笼包", "牛肉面", "炸酱面", "皮蛋瘦肉粥"]
SHOPS = ["牛奶", "鸡蛋", "面包", "苹果", "牙膏", "大米", "酱油", "纸巾", "香蕉", "酸奶"]
ROSTER = ["张伟", "李娜", "王强", "赵敏", "刘洋", "周杰", "吴迪", "郑爽"]


def _py(ch: str) -> str:
    return lazy_pinyin(ch)[0]


_BY_PY: dict[str, list[str]] = {}
for _ch in BANK:
    _BY_PY.setdefault(_py(_ch), []).append(_ch)


def homo(ch: str, rng: random.Random) -> str | None:
    cands = [x for x in _BY_PY.get(_py(ch), []) if x != ch and x not in BANNED_NAME_CHARS]
    if not cands:
        return None
    return rng.choice(cands)


def _end(rng: random.Random) -> str:
    return "。" if rng.random() < 0.45 else ""


def _clarify_pair(surface: str, gold: str, desc_src: str, desc_gold: str, rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(6)
    if kind == 0:
        return f"{surface}是{desc_src}的{surface}", f"{gold}是{desc_gold}的{gold}"
    if kind == 1:
        return f"{surface}，{desc_src}的{surface}", f"{gold}，{desc_gold}的{gold}"
    if kind == 2:
        return f"{surface}字是{desc_src}的{surface}", f"{gold}字是{desc_gold}的{gold}"
    if kind == 3:
        return f"{surface}写成{desc_src}的{surface}", f"{gold}写成{desc_gold}的{gold}"
    if kind == 4:
        return f"{desc_src}那个{surface}", f"{desc_gold}那个{gold}"
    return f"不是别字，是{desc_src}的{surface}", f"不是别字，是{desc_gold}的{gold}"


def _frame(scene: str, src_ent: str, gold_ent: str, src_c: str, gold_c: str, rng: random.Random) -> tuple[str, str]:
    p = _end(rng)
    if scene == "person":
        options = [
            (f"我叫{src_ent}，{src_c}{p}", f"我叫{gold_ent}，{gold_c}{p}"),
            (f"我是{src_ent}。{src_c}{p}", f"我是{gold_ent}。{gold_c}{p}"),
            (f"我的名字是{src_ent}，{src_c}{p}", f"我的名字是{gold_ent}，{gold_c}{p}"),
            (f"叫我{src_ent}。{src_c}{p}", f"叫我{gold_ent}。{gold_c}{p}"),
            (f"大家叫我{src_ent}，{src_c}{p}", f"大家叫我{gold_ent}，{gold_c}{p}"),
        ]
    elif scene == "place":
        options = [
            (f"我住在{src_ent}，{src_c}{p}", f"我住在{gold_ent}，{gold_c}{p}"),
            (f"导航到{src_ent}。{src_c}{p}", f"导航到{gold_ent}。{gold_c}{p}"),
            (f"地铁站是{src_ent}，{src_c}{p}", f"地铁站是{gold_ent}，{gold_c}{p}"),
            (f"地址写{src_ent}。{src_c}{p}", f"地址写{gold_ent}。{gold_c}{p}"),
        ]
    elif scene == "food":
        options = [
            (f"来一份{src_ent}，{src_c}{p}", f"来一份{gold_ent}，{gold_c}{p}"),
            (f"菜名是{src_ent}。{src_c}{p}", f"菜名是{gold_ent}。{gold_c}{p}"),
            (f"这家做{src_ent}，{src_c}{p}", f"这家做{gold_ent}，{gold_c}{p}"),
        ]
    elif scene == "org":
        options = [
            (f"公司叫{src_ent}，{src_c}{p}", f"公司叫{gold_ent}，{gold_c}{p}"),
            (f"学校是{src_ent}。{src_c}{p}", f"学校是{gold_ent}。{gold_c}{p}"),
            (f"店名是{src_ent}，{src_c}{p}", f"店名是{gold_ent}，{gold_c}{p}"),
            (f"品牌叫{src_ent}。{src_c}{p}", f"品牌叫{gold_ent}。{gold_c}{p}"),
        ]
    else:
        options = [
            (f"这个字写成{src_ent}，{src_c}{p}", f"这个字写成{gold_ent}，{gold_c}{p}"),
            (f"这里的{src_ent}，{src_c}{p}", f"这里的{gold_ent}，{gold_c}{p}"),
        ]
    return rng.choice(options)


def _ok_name(ent: str) -> bool:
    if any(ch in BANNED_NAME_CHARS for ch in ent):
        return False
    if len(ent) >= 2 and ent[-1] == ent[-2]:
        return False
    return True


def _homos(ch: str) -> list[str]:
    return [x for x in _BY_PY.get(_py(ch), []) if x != ch and x not in BANNED_NAME_CHARS]


def _clause_surface(name_surface: str, gold: str, rng: random.Random) -> str:
    """Name glyph and the glyph inside the clarification can differ."""
    if rng.random() < 0.65:
        return name_surface
    cands = [gold, name_surface, *_homos(gold)]
    return rng.choice(cands)


def sample_word_edit(rng: random.Random, scenes: list[str]) -> tuple[str, str] | None:
    tgt, word = rng.choice(WORDS)
    if tgt not in word or tgt in BANNED_NAME_CHARS:
        return None
    surface = homo(tgt, rng)
    if not surface or surface == tgt:
        return None
    scene = rng.choice(scenes)
    if scene == "bare":
        src_ent, gold_ent = surface, tgt
    elif scene == "person":
        sur = rng.choice(SURNAMES)
        extra = rng.choice(["", rng.choice("小文雨子嘉欣晨")])
        src_ent, gold_ent = sur + extra + surface, sur + extra + tgt
    else:
        extra = rng.choice(["华", "东", "新", "大", "小", ""])
        src_ent, gold_ent = extra + surface, extra + tgt
    if not _ok_name(src_ent) or not _ok_name(gold_ent):
        return None
    clause = _clause_surface(surface, tgt, rng)
    src_c, gold_c = _clarify_pair(clause, tgt, word, word, rng)
    return _frame(scene, src_ent, gold_ent, src_c, gold_c, rng)


def sample_radical_edit(rng: random.Random, scenes: list[str], pool: list[tuple[str, str, str, str]]) -> tuple[str, str] | None:
    spoken_rad, surface, gold, gold_rad = rng.choice(pool)
    if surface in BANNED_NAME_CHARS or gold in BANNED_NAME_CHARS:
        return None
    if surface == gold and spoken_rad == gold_rad:
        return None
    scene = rng.choice(scenes)
    if scene == "bare":
        src_ent, gold_ent = surface, gold
    elif scene == "person":
        sur = rng.choice(SURNAMES)
        extra = rng.choice(["", rng.choice("小文雨子嘉欣晨")])
        src_ent, gold_ent = sur + extra + surface, sur + extra + gold
    else:
        extra = rng.choice(["华", "东", "新", "大", "小", ""])
        src_ent, gold_ent = extra + surface, extra + gold
    if not _ok_name(src_ent) or not _ok_name(gold_ent):
        return None
    clause = _clause_surface(surface, gold, rng)
    src_c, gold_c = _clarify_pair(clause, gold, spoken_rad, gold_rad, rng)
    return _frame(scene, src_ent, gold_ent, src_c, gold_c, rng)


# slot = (name_surface, gold, clause_surface, desc_src, desc_gold)
Slot = tuple[str, str, str, str, str]
DUAL_SCENES = ["person", "person", "person", "place", "food", "org", "bare"]


def _word_slot(rng: random.Random, keep: bool, force_ming: bool = False) -> Slot | None:
    if keep and force_ming:
        tgt, word = rng.choice([("明", "明日"), ("明", "明天")])
        return tgt, tgt, tgt, word, word
    tgt, word = rng.choice(WORDS)
    if tgt not in word or tgt in BANNED_NAME_CHARS:
        return None
    if keep:
        return tgt, tgt, tgt, word, word
    surface = homo(tgt, rng)
    if not surface or surface == tgt:
        return None
    return surface, tgt, _clause_surface(surface, tgt, rng), word, word


def _rad_slot(rng: random.Random, pool: list[tuple[str, str, str, str]], keep: bool) -> Slot | None:
    spoken, surface, gold, gold_rad = rng.choice(pool)
    if surface in BANNED_NAME_CHARS or gold in BANNED_NAME_CHARS:
        return None
    if keep:
        return gold, gold, gold, gold_rad, gold_rad
    if surface == gold and spoken == gold_rad:
        return None
    return surface, gold, _clause_surface(surface, gold, rng), spoken, gold_rad


def _forced_ming_yang(rng: random.Random, pool: list[tuple[str, str, str, str]]) -> tuple[Slot, Slot] | None:
    """明/名 plus 提手旁→扬, without fixing the surname to 陈."""
    aliases = ("提手旁", "铁手旁", "剔手旁", "体手旁", "提首旁")
    cands = [r for r in pool if r[0] in aliases and r[2] == "扬" and r[1] != "扬"]
    if not cands:
        return None
    spoken, surface, gold, gold_rad = rng.choice(cands)
    # Name may already show 名 while the other slot is still a 扬 error, or both err.
    if rng.random() < 0.45:
        word: Slot = ("明", "名", "名", "名字", "名字")
    else:
        word = ("明", "名", "明", "名字", "名字")
    rad: Slot = (surface, gold, _clause_surface(surface, gold, rng), spoken, gold_rad)
    if rng.random() < 0.5:
        return word, rad
    return rad, word


def _slots_ok(a: Slot | None, b: Slot | None) -> bool:
    if a is None or b is None:
        return False
    own_a = {a[0], a[1], a[2]}
    own_b = {b[0], b[1], b[2]}
    if own_a & own_b:
        return False
    blob_a = a[3] + a[4]
    blob_b = b[3] + b[4]
    if any(ch in blob_a for ch in own_b):
        return False
    if any(ch in blob_b for ch in own_a):
        return False
    return True


def _two_entity(scene: str, a: Slot, b: Slot, rng: random.Random) -> tuple[str, str] | None:
    blocked = {a[0], a[1], b[0], b[1]}
    if scene == "person":
        surs = [s for s in SURNAMES if s not in blocked]
        sur = rng.choice(surs)
        src, gold = sur + a[0] + b[0], sur + a[1] + b[1]
    elif scene == "bare":
        src, gold = a[0] + b[0], a[1] + b[1]
    else:
        extras = [e for e in ["华", "东", "新", "大", "小", ""] if not e or e not in blocked]
        extra = rng.choice(extras)
        src, gold = extra + a[0] + b[0], extra + a[1] + b[1]
    if not _ok_name(src) or not _ok_name(gold):
        return None
    return src, gold


def _two_frame(scene: str, src_ent: str, gold_ent: str, c1s: str, c1g: str, c2s: str, c2g: str, rng: random.Random) -> tuple[str, str]:
    join = rng.choice(["，", "。"])
    end = _end(rng)
    if scene == "person":
        leads = ["我叫{e}，", "我是{e}，", "我是{e}。", "我的名字是{e}，", "叫我{e}，", "大家叫我{e}。"]
    elif scene == "place":
        leads = ["我住在{e}，", "导航到{e}，", "地址写{e}。", "地铁站是{e}，"]
    elif scene == "food":
        leads = ["来一份{e}，", "菜名是{e}。", "这家做{e}，"]
    elif scene == "org":
        leads = ["公司叫{e}，", "学校是{e}。", "店名是{e}，", "品牌叫{e}。"]
    else:
        leads = ["这两个字是{e}，", "写成{e}，", "这里写{e}。"]
    lead = rng.choice(leads)
    src = lead.format(e=src_ent) + c1s + join + c2s + end
    gold = lead.format(e=gold_ent) + c1g + join + c2g + end
    return src, gold


def sample_dual(rng: random.Random, pool: list[tuple[str, str, str, str]], mode: str) -> tuple[str, str] | None:
    """mode: edit (both change), contrast (one stays), keep (both stay)."""
    for _ in range(16):
        if mode == "edit" and rng.random() < 0.35:
            pair = _forced_ming_yang(rng, pool)
            if pair is None:
                continue
            a, b = pair
        elif mode == "contrast":
            keep_first = rng.random() < 0.5
            force_ming = rng.random() < 0.6
            slots: list[Slot | None] = []
            for is_keep in (keep_first, not keep_first):
                if is_keep and force_ming:
                    slots.append(_word_slot(rng, True, force_ming=True))
                elif is_keep:
                    if rng.random() < 0.7:
                        slots.append(_word_slot(rng, True))
                    else:
                        slots.append(_rad_slot(rng, pool, True))
                elif rng.random() < 0.55:
                    slots.append(_word_slot(rng, False))
                else:
                    slots.append(_rad_slot(rng, pool, False))
            a, b = slots
        elif mode == "keep":
            a = _word_slot(rng, True, force_ming=rng.random() < 0.3)
            b = _rad_slot(rng, pool, True) if rng.random() < 0.5 else _word_slot(rng, True)
        else:
            roll = rng.randrange(3)
            if roll == 0:
                a, b = _word_slot(rng, False), _word_slot(rng, False)
            elif roll == 1:
                a, b = _rad_slot(rng, pool, False), _rad_slot(rng, pool, False)
            elif rng.random() < 0.5:
                a, b = _word_slot(rng, False), _rad_slot(rng, pool, False)
            else:
                a, b = _rad_slot(rng, pool, False), _word_slot(rng, False)
        if not _slots_ok(a, b):
            continue
        scene = rng.choice(DUAL_SCENES)
        ent = _two_entity(scene, a, b, rng)
        if ent is None:
            continue
        c1s, c1g = _clarify_pair(a[2], a[1], a[3], a[4], rng)
        c2s, c2g = _clarify_pair(b[2], b[1], b[3], b[4], rng)
        src, gold = _two_frame(scene, ent[0], ent[1], c1s, c1g, c2s, c2g, rng)
        changed = src != gold
        if mode == "keep" and changed:
            continue
        if mode != "keep" and not changed:
            continue
        return src, gold
    return None


def sample_already(rng: random.Random, scenes: list[str], pool: list[tuple[str, str, str, str]]) -> tuple[str, str] | None:
    if rng.random() < 0.4:
        return sample_dual(rng, pool, "keep")
    if rng.random() < 0.5:
        tgt, word = rng.choice(WORDS)
        if tgt not in word or tgt in BANNED_NAME_CHARS:
            return None
        surface, gold, desc_s, desc_g = tgt, tgt, word, word
    else:
        _rad, _surface, gold, grad = rng.choice(RADICALS)
        if gold in BANNED_NAME_CHARS:
            return None
        surface, desc_s, desc_g = gold, grad, grad
    scene = rng.choice(scenes)
    if scene == "person":
        sur = rng.choice(SURNAMES)
        ent = sur + rng.choice(["", "小"]) + gold
    elif scene == "bare":
        ent = gold
    else:
        ent = rng.choice(["华", "东", "新", ""]) + gold
    if not _ok_name(ent):
        return None
    src_c, gold_c = _clarify_pair(surface, gold, desc_s, desc_g, rng)
    src, dst = _frame(scene, ent, ent, src_c, gold_c, rng)
    if src != dst:
        return None
    return src, dst


def sample_plain(rng: random.Random) -> tuple[str, str]:
    time = rng.choice(["今天", "明天", "下午", "晚上", "周末", "周一"])
    who = rng.choice(["我", "我们", "他", "大家"])
    act = rng.choice(["开会", "买菜", "回家", "发邮件", "吃饭", "休息", "看文件", "坐地铁", "去公司", "订票"])
    place = rng.choice(["在公司", "在家", "在学校", "在会议室", "在超市", ""])
    tail = rng.choice(["。", ""])
    text = f"{time}{who}{place}{act}{tail}"
    if "是" in text and "的" in text:
        return sample_plain(rng)
    return text, text


def _row(src: str, gold: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": INSTRUCTION.format(asr=src)},
            {"role": "assistant", "content": gold},
        ]
    }


def _accept(src: str, gold: str, for_train: bool) -> bool:
    if "陈名扬" in src or "陈名扬" in gold:
        return not for_train and (src, gold) in MUST_VAL
    if for_train and any(src == a and gold == b for a, b in MUST_VAL):
        return False
    return True


SCENES = ["person", "place", "food", "org", "bare"]


def build_split(n: int, rng: random.Random, for_train: bool, radical_pool: list[tuple[str, str, str, str]]) -> tuple[list[dict], dict[str, int]]:
    # 70% edits: single 组词 / single 偏旁 / one sentence with two clarifications.
    n_word = int(n * 0.28)
    n_rad = int(n * 0.20)
    n_dual_edit = int(n * 0.10)
    n_dual_contrast = int(n * 0.12)
    n_ok = int(n * 0.15)
    n_plain = n - n_word - n_rad - n_dual_edit - n_dual_contrast - n_ok
    rows: list[dict] = []
    seen: set[str] = set()
    counts = {"word": 0, "rad": 0, "dual_edit": 0, "dual_contrast": 0, "ok": 0, "plain": 0}

    def add(pair: tuple[str, str] | None, bucket: str) -> bool:
        if pair is None:
            return False
        src, gold = pair
        if src in seen or not _accept(src, gold, for_train):
            return False
        seen.add(src)
        rows.append(_row(src, gold))
        counts[bucket] += 1
        return True

    quotas = [
        (n_word, "word", lambda: sample_word_edit(rng, SCENES)),
        (n_rad, "rad", lambda: sample_radical_edit(rng, SCENES, radical_pool)),
        (n_dual_edit, "dual_edit", lambda: sample_dual(rng, radical_pool, "edit")),
        (n_dual_contrast, "dual_contrast", lambda: sample_dual(rng, radical_pool, "contrast")),
        (n_ok, "ok", lambda: sample_already(rng, SCENES, radical_pool)),
        (n_plain, "plain", lambda: sample_plain(rng)),
    ]
    for need, bucket, fn in quotas:
        guard = 0
        got = 0
        while got < need and guard < need * 200:
            guard += 1
            if add(fn(), bucket):
                got += 1
        if got < need:
            raise RuntimeError(f"only generated {got}/{need} for {bucket}")
    rng.shuffle(rows)
    return rows, counts


def _retarget_instruction(content: str) -> str:
    if "【原文】" not in content:
        raise ValueError("missing 原文")
    asr = content.split("【原文】", 1)[1]
    return INSTRUCTION.format(asr=asr)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _holdout_src() -> set[str]:
    return {a for a, _ in MUST_VAL + LIST_MUST + PARA_MUST + NEW_MUST}


def sample_ordered_list(rng: random.Random) -> tuple[str, str]:
    k = rng.choice([2, 3, 4])
    items = rng.sample(TASKS, k)
    unit = "点" if rng.random() < 0.35 else "条"
    parts: list[str] = []
    for i, item in enumerate(items):
        if i > 0 and rng.random() < 0.5:
            marker = "下一条" if unit == "条" else "下一点"
        else:
            marker = f"第{_CN[i]}{unit}"
        style = rng.randrange(3)
        if style == 0:
            parts.append(f"{marker}{item}")
        elif style == 1:
            parts.append(f"{marker}，{item}")
        else:
            parts.append(f"{marker}是{item}")
    body = rng.choice(["，", "。", "、"]).join(parts)
    lead = rng.choice(["", "帮我记一下，", "待办是，", "安排如下，"])
    src = lead + body
    if rng.random() < 0.45 and not src.endswith("。"):
        src += "。"
    lines = [f"{i}. {item}" for i, item in enumerate(items, 1)]
    gold = "\n".join(lines)
    if lead:
        gold = lead.rstrip("，") + "\n" + gold
    return src, gold


def sample_unordered_list(rng: random.Random) -> tuple[str, str]:
    kind = rng.choice(["dish", "shop", "name"])
    if kind == "dish":
        pool, leads = DISHES, ["报个菜名：", "菜名是", "今天的菜有", "点菜：", "菜单上有"]
    elif kind == "shop":
        pool, leads = SHOPS, ["要买", "购物清单：", "买这些：", "采购："]
    else:
        pool, leads = ROSTER, ["名单是", "分别是", "有这些：", "参加的人有"]
    items = rng.sample(pool, rng.choice([2, 3, 4, 5]))
    sep = rng.choice(["、", "，", "，"])
    body = sep.join(items)
    lead = rng.choice(leads)
    src = lead + body
    if rng.random() < 0.4:
        src += "。"
    gold = lead + items[0] + "\n" + "\n".join(items[1:])
    return src, gold


def sample_list_negative(rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(4)
    if kind == 0:
        text = rng.choice([
            "合同第一条不作修改。",
            "今天第一条新闻讲的是天气。",
            "这条路第一条岔路往左。",
            "法规第一条仍然有效。",
            "第一条消息我已经看到了。",
        ])
        return text, text
    if kind == 1:
        text = rng.choice([
            "我去公司，然后回家。",
            "今天下午开会，晚上回家吃饭。",
            "先把文件发出去，再打电话。",
            "明天出差，后天回来。",
            "菜已经点了，等一会儿上。",
        ])
        return text, text
    if kind == 2:
        items = rng.sample(TASKS, rng.choice([2, 3]))
        text = "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
        return text, text
    items = rng.sample(DISHES, rng.choice([2, 3]))
    text = "\n".join(items)
    return text, text


def build_list_rows(rng: random.Random, n_ord: int, n_unord: int, n_neg: int, for_train: bool) -> tuple[list[dict], dict[str, int]]:
    holdout = _holdout_src()
    seen: set[str] = set()
    rows: list[dict] = []
    counts = {"ordered": 0, "unordered": 0, "list_neg": 0}

    def add(pair: tuple[str, str] | None, bucket: str) -> bool:
        if pair is None:
            return False
        src, gold = pair
        if src in seen or src in holdout:
            return False
        if for_train and ("陈名扬" in src or "陈名扬" in gold):
            return False
        seen.add(src)
        rows.append(_row(src, gold))
        counts[bucket] += 1
        return True

    quotas = [
        (n_ord, "ordered", lambda: sample_ordered_list(rng)),
        (n_unord, "unordered", lambda: sample_unordered_list(rng)),
        (n_neg, "list_neg", lambda: sample_list_negative(rng)),
    ]
    for need, bucket, fn in quotas:
        guard = 0
        got = 0
        while got < need and guard < need * 80:
            guard += 1
            if add(fn(), bucket):
                got += 1
        if got < need:
            raise RuntimeError(f"only generated {got}/{need} for {bucket}")
    return rows, counts


def _matter_span(rng: random.Random, matter: list[str]) -> str:
    k = rng.choice([1, 2])
    k = min(k, len(matter))
    start = rng.randrange(0, len(matter) - k + 1)
    return "".join(matter[start : start + k])


def _pick_blocks(rng: random.Random, n: int) -> list[str]:
    chosen = rng.sample(MATTERS, n)
    return [_matter_span(rng, matter) for matter in chosen]


def sample_spoken_paragraph(rng: random.Random) -> tuple[str, str]:
    blocks = _pick_blocks(rng, rng.choice([2, 3]))
    parts: list[str] = []
    for i, text in enumerate(blocks):
        if i > 0 and rng.random() < 0.5:
            marker = "下一段"
        else:
            marker = f"第{_CN[i]}段"
        style = rng.randrange(3)
        if style == 0:
            parts.append(f"{marker}{text}")
        elif style == 1:
            parts.append(f"{marker}，{text}")
        else:
            parts.append(f"{marker}是{text}")
    src = "".join(parts)
    if not src.endswith("。"):
        src += "。"
    return src, "\n\n".join(blocks)


def sample_topic_paragraph(rng: random.Random) -> tuple[str, str]:
    blocks = _pick_blocks(rng, rng.choice([2, 3]))
    src = blocks[0]
    for block in blocks[1:]:
        joiner = rng.choice(["另外", "还有，", "接下来，", ""])
        src += joiner + block
    return src, "\n\n".join(blocks)


def sample_paragraph_negative(rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(3)
    if kind == 0:
        matter = rng.choice(MATTERS)
        k = min(rng.choice([2, 3]), len(matter))
        text = "".join(matter[:k])
        return text, text
    if kind == 1:
        text = rng.choice([
            "今天下午开会，晚上把纪要写完。",
            "我去公司，然后回家。",
            "先把文件发出去，再打电话。",
            "明天出差，后天回来。",
            "今天天气不错，适合出门散步。",
        ])
        return text, text
    text = "\n\n".join(_pick_blocks(rng, rng.choice([2, 3])))
    return text, text


def build_paragraph_rows(rng: random.Random, n_spoken: int, n_topic: int, n_neg: int, for_train: bool) -> tuple[list[dict], dict[str, int]]:
    holdout = _holdout_src()
    seen: set[str] = set()
    rows: list[dict] = []
    counts = {"spoken_para": 0, "topic_para": 0, "para_neg": 0}

    def add(pair: tuple[str, str] | None, bucket: str) -> bool:
        if pair is None:
            return False
        src, gold = pair
        if src in seen or src in holdout:
            return False
        if "第" in src and ("条" in src or "点" in src):
            return False
        if for_train and ("陈名扬" in src or "陈名扬" in gold):
            return False
        seen.add(src)
        rows.append(_row(src, gold))
        counts[bucket] += 1
        return True

    quotas = [
        (n_spoken, "spoken_para", lambda: sample_spoken_paragraph(rng)),
        (n_topic, "topic_para", lambda: sample_topic_paragraph(rng)),
        (n_neg, "para_neg", lambda: sample_paragraph_negative(rng)),
    ]
    for need, bucket, fn in quotas:
        guard = 0
        got = 0
        while got < need and guard < need * 80:
            guard += 1
            if add(fn(), bucket):
                got += 1
        if got < need:
            raise RuntimeError(f"only generated {got}/{need} for {bucket}")
    return rows, counts


def _write_split(out_dir: Path, train_rows: list[dict], val_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in ("train", train_rows), ("val", val_rows):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                blob = json.dumps(row, ensure_ascii=False)
                if name == "train" and "陈名扬" in blob:
                    raise SystemExit("train leaked 陈名扬")
                f.write(blob + "\n")
        print(f"{name}: {len(rows)} -> {path}")


def _wrong_char(rng: random.Random, avoid: set[str]) -> str:
    pool = [ch for ch in WRONG_CHARS if ch not in avoid]
    return rng.choice(pool)


def to_cn(n: int) -> str:
    if n < 10:
        return _D[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        text = "十" if tens == 1 else _D[tens] + "十"
        if ones:
            text += _D[ones]
        return text
    hundreds, rest = divmod(n, 100)
    text = _D[hundreds] + "百"
    if rest == 0:
        return text
    if rest < 10:
        return text + "零" + _D[rest]
    tens, ones = divmod(rest, 10)
    text += "一十" if tens == 1 else _D[tens] + "十"
    if ones:
        text += _D[ones]
    return text


def sample_component(rng: random.Random) -> tuple[str, str]:
    gold, parts = rng.choice(SPLITS)
    avoid = set(gold) | set(parts)
    left = gold if rng.random() < 0.4 else _wrong_char(rng, avoid)
    right = gold if rng.random() < 0.4 else _wrong_char(rng, avoid)
    frame = rng.randrange(3)
    if frame == 0:
        return f"我姓{left}，{left}是{parts}{right}", f"我姓{gold}，{gold}是{parts}{gold}"
    if frame == 1:
        return f"{left}是{parts}{right}", f"{gold}是{parts}{gold}"
    return f"{left}，{parts}{right}", f"{gold}，{parts}{gold}"


def sample_radical_override(rng: random.Random) -> tuple[str, str]:
    radical, gold = rng.choice(RADICAL_GOLD)
    avoid = set(gold) | set(radical)
    if rng.random() < 0.3:
        left = right = gold
    else:
        left = _wrong_char(rng, avoid)
        right = _wrong_char(rng, avoid)
    return f"{left}是{radical}的{right}", f"{gold}是{radical}的{gold}"


def _minute_spoken(m: int, with_fen: bool) -> str:
    if m == 0:
        return "整"
    if m < 10:
        return _D[m] + "分"
    tens, ones = divmod(m, 10)
    text = _D[tens] + "十"
    if ones:
        text += _D[ones]
    if with_fen:
        text += "分"
    return text


def sample_time(rng: random.Random) -> tuple[str, str]:
    hour = rng.randint(1, 12)
    minute = rng.randint(0, 59)
    spoken = _minute_spoken(minute, minute < 10 or rng.random() < 0.5)
    lead = rng.choice(["现在", "下午", "会议在", "大概", ""])
    return f"{lead}{to_cn(hour)}点{spoken}", f"{lead}{hour}:{minute:02d}"


def sample_math(rng: random.Random) -> tuple[str, str]:
    op_cn, op_sym, kind = rng.choice([
        ("加", "+", "add"),
        ("减", "-", "sub"),
        ("乘", "×", "mul"),
        ("除", "÷", "div"),
    ])
    if kind == "div":
        b = rng.randint(1, 9)
        result = rng.randint(0, 9)
        a = b * result
    elif kind == "sub":
        a = rng.randint(0, 20)
        b = rng.randint(0, a)
        result = a - b
    elif kind == "mul":
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        result = a * b
    else:
        a = rng.randint(0, 20)
        b = rng.randint(0, 20)
        result = a + b
    lead = rng.choice(["", "算一下，", "结果是，"])
    src = f"{lead}{to_cn(a)}{op_cn}{to_cn(b)}等于{to_cn(result)}"
    gold = f"{lead}{a} {op_sym} {b} = {result}"
    return src, gold


def sample_plain_number(rng: random.Random) -> tuple[str, str]:
    n = rng.choice([7, 15, 20, 36, 48, 100, 108, 123, 256, rng.randint(0, 199)])
    frame = rng.choice([
        ("一共{cn}块", "一共{n}块"),
        ("编号是{cn}", "编号是{n}"),
        ("订了{cn}份", "订了{n}份"),
    ])
    return frame[0].format(cn=to_cn(n)), frame[1].format(n=n)


def sample_number_negative(rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(4)
    if kind == 0:
        verb = rng.choice(["迟到", "忘记", "着急", "乱跑", "大声说话", "走错路", "丢东西", "答应太快"])
        obj = rng.choice(["", "这件事", "钥匙", "文件"])
        text = f"千万不要{verb}{obj}。"
    elif kind == 1:
        place = rng.choice(["左边", "右边", "名字里", "这个字", "姓氏上", "抬头上"])
        text = f"{place}是三点水。"
    elif kind == 2:
        lead = rng.choice(["", "然后", "接着", "请", "记得"])
        text = f"{lead}下一条{rng.choice(TASKS)}。"
    else:
        bit = rng.choice(["先开会", "讨论合同", "写纪要", "再回家", "介绍背景", "确认时间", "总结一下", "安排座位"])
        text = f"第一段{bit}。"
    return text, text


def build_norm_rows(rng: random.Random, for_train: bool) -> tuple[list[dict], dict[str, int]]:
    if for_train:
        quotas = [
            (800, "component", lambda: sample_component(rng)),
            (800, "radical_override", lambda: sample_radical_override(rng)),
            (280, "time", lambda: sample_time(rng)),
            (250, "math", lambda: sample_math(rng)),
            (150, "plain_number", lambda: sample_plain_number(rng)),
            (120, "number_neg", lambda: sample_number_negative(rng)),
        ]
    else:
        quotas = [
            (20, "component", lambda: sample_component(rng)),
            (20, "radical_override", lambda: sample_radical_override(rng)),
            (12, "time", lambda: sample_time(rng)),
            (10, "math", lambda: sample_math(rng)),
            (8, "plain_number", lambda: sample_plain_number(rng)),
            (8, "number_neg", lambda: sample_number_negative(rng)),
        ]
    holdout = _holdout_src()
    seen: set[str] = set()
    rows: list[dict] = []
    counts = {name: 0 for _, name, _ in quotas}

    def add(pair: tuple[str, str] | None, bucket: str) -> bool:
        if pair is None:
            return False
        src, gold = pair
        if src in seen or src in holdout:
            return False
        if for_train and ("陈名扬" in src or "陈名扬" in gold):
            return False
        seen.add(src)
        rows.append(_row(src, gold))
        counts[bucket] += 1
        return True

    for need, bucket, fn in quotas:
        guard = 0
        got = 0
        while got < need and guard < need * 80:
            guard += 1
            if add(fn(), bucket):
                got += 1
        if got < need:
            raise RuntimeError(f"only generated {got}/{need} for {bucket}")
    return rows, counts


def build_v5(v4_dir: Path, out_dir: Path, rng: random.Random) -> None:
    holdout = _holdout_src()
    train_rows = _load_jsonl(v4_dir / "train.jsonl")
    val_rows = _load_jsonl(v4_dir / "val.jsonl")
    kept = []
    for row in train_rows:
        row["messages"][0]["content"] = _retarget_instruction(row["messages"][0]["content"])
        src = row["messages"][0]["content"].split("【原文】", 1)[1]
        if src in holdout:
            continue
        kept.append(row)
    train_rows = kept
    for row in val_rows:
        row["messages"][0]["content"] = _retarget_instruction(row["messages"][0]["content"])
    extra_train, train_counts = build_norm_rows(rng, True)
    extra_val, val_counts = build_norm_rows(rng, False)
    train_rows.extend(extra_train)
    rng.shuffle(train_rows)
    must = [_row(a, b) for a, b in MUST_VAL + LIST_MUST + PARA_MUST + NEW_MUST]
    have: set[str] = set()
    merged_val = []
    for row in must + extra_val + val_rows:
        src = row["messages"][0]["content"].split("【原文】", 1)[1]
        if src in have:
            continue
        have.add(src)
        merged_val.append(row)
    _write_split(out_dir, train_rows, merged_val)
    print("v5 train", train_counts)
    print("v5 val", val_counts)


def build_v4(v3_dir: Path, out_dir: Path, rng: random.Random) -> None:
    holdout = _holdout_src()
    train_rows = _load_jsonl(v3_dir / "train.jsonl")
    val_rows = _load_jsonl(v3_dir / "val.jsonl")
    kept = []
    for row in train_rows:
        row["messages"][0]["content"] = _retarget_instruction(row["messages"][0]["content"])
        src = row["messages"][0]["content"].split("【原文】", 1)[1]
        if src in holdout:
            continue
        kept.append(row)
    train_rows = kept
    for row in val_rows:
        row["messages"][0]["content"] = _retarget_instruction(row["messages"][0]["content"])
    para_train, train_counts = build_paragraph_rows(rng, 800, 800, 600, True)
    para_val, val_counts = build_paragraph_rows(rng, 30, 30, 20, False)
    train_rows.extend(para_train)
    rng.shuffle(train_rows)
    must = [_row(a, b) for a, b in MUST_VAL + LIST_MUST + PARA_MUST]
    have: set[str] = set()
    merged_val = []
    for row in must + para_val + val_rows:
        src = row["messages"][0]["content"].split("【原文】", 1)[1]
        if src in have:
            continue
        have.add(src)
        merged_val.append(row)
    _write_split(out_dir, train_rows, merged_val)
    print("paragraph train", train_counts)
    print("paragraph val", val_counts)


def build_v3(v2_dir: Path, out_dir: Path, rng: random.Random) -> None:
    train_rows = _load_jsonl(v2_dir / "train.jsonl")
    val_rows = _load_jsonl(v2_dir / "val.jsonl")
    for row in train_rows + val_rows:
        row["messages"][0]["content"] = _retarget_instruction(row["messages"][0]["content"])
    list_train, train_counts = build_list_rows(rng, 1000, 1000, 400, True)
    list_val, val_counts = build_list_rows(rng, 40, 40, 20, False)
    train_rows.extend(list_train)
    rng.shuffle(train_rows)
    must = [_row(a, b) for a, b in MUST_VAL + LIST_MUST]
    have = set()
    merged_val = []
    for row in must + list_val + val_rows:
        src = row["messages"][0]["content"].split("【原文】", 1)[1]
        if src in have:
            continue
        have.add(src)
        merged_val.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in ("train", train_rows), ("val", merged_val):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                blob = json.dumps(row, ensure_ascii=False)
                if name == "train" and "陈名扬" in blob:
                    raise SystemExit("train leaked 陈名扬")
                f.write(blob + "\n")
        print(f"{name}: {len(rows)} -> {path}")
    print("list train", train_counts)
    print("list val", val_counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--v3-from", type=Path, default=None, help="rewrite this v2 dir and append list rows")
    ap.add_argument("--v4-from", type=Path, default=None, help="rewrite this v3 dir and append paragraph rows")
    ap.add_argument("--v5-from", type=Path, default=None, help="rewrite this v4 dir and append split, radical, and number rows")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.v5_from is not None:
        build_v5(args.v5_from, args.out, rng)
        return
    if args.v4_from is not None:
        build_v4(args.v4_from, args.out, rng)
        return
    if args.v3_from is not None:
        build_v3(args.v3_from, args.out, rng)
        return

    pool = list(RADICALS)
    train_rows, train_counts = build_split(args.train, rng, True, pool)
    val_rows = [_row(a, b) for a, b in MUST_VAL]
    extra, val_counts = build_split(max(0, args.val - len(val_rows)), rng, False, pool)
    val_rows.extend(extra)
    print("train mix", train_counts)
    print("val mix", val_counts)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in ("train", train_rows), ("val", val_rows):
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                blob = json.dumps(row, ensure_ascii=False)
                if name == "train" and "陈名扬" in blob:
                    raise SystemExit("train leaked 陈名扬")
                f.write(blob + "\n")
        print(f"{name}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
