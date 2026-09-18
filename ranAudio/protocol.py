"""§4 stream tags + CER / parroting metrics. Not the v1.1 JSON judge."""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
import torch.nn.functional as F

MODE_SPEAK = "<mode:speak>"
MODE_ROUTE = "<mode:route>"

_MODE_RE = re.compile(r"<mode:(speak|route)>")
_ASR_RE = re.compile(r"<asr>(.*?)</asr>", re.DOTALL)
_FILLER_RE = re.compile(r"<filler>(.*?)</filler>", re.DOTALL)
_REPLY_RE = re.compile(r"<reply>(.*?)</reply>", re.DOTALL)
_DIGIT_RE = re.compile(r"\d+")

ROUTE_SYSTEM = (
    "你是车机语音助手。先输出路由标签，再输出对应内容。"
    "需要调用车机工具（蓝牙、导航、空调、音量、车窗、电话、音乐、座椅加热、大灯等）时："
    f"{MODE_ROUTE}<asr>转写</asr><filler>垫话</filler>。"
    "闲聊或普通问答："
    f"{MODE_SPEAK}<reply>回复</reply>。"
    "不要输出工具名、参数或 function call。"
)

ASR_SYSTEM = "请把用户语音转成中文文本，只输出转写。"

REPLY_SYSTEM = (
    "你是普通中文助手。根据用户语音直接回复。"
    "不要调用工具，不要输出 JSON 或 XML。"
)

ROUTE_ONLY_SYSTEM = (
    "只判断是否需要调用车机工具。只输出 "
    f"{MODE_ROUTE} 或 {MODE_SPEAK}，不要转写，不要回复。"
)


def format_speak(reply: str) -> str:
    return f"{MODE_SPEAK}<reply>{reply}</reply>"


def format_route(asr: str, filler: str = "好的") -> str:
    return f"{MODE_ROUTE}<asr>{asr}</asr><filler>{filler}</filler>"


def parse_stream(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    mode_m = _MODE_RE.search(raw)
    asr_m = _ASR_RE.search(raw)
    filler_m = _FILLER_RE.search(raw)
    reply_m = _REPLY_RE.search(raw)
    mode = mode_m.group(1) if mode_m else None
    if mode is None:
        if asr_m and not reply_m:
            mode = "route"
        elif reply_m and not asr_m:
            mode = "speak"
    return {
        "mode": mode,
        "asr": (asr_m.group(1) if asr_m else "").strip(),
        "filler": (filler_m.group(1) if filler_m else "").strip(),
        "reply": (reply_m.group(1) if reply_m else "").strip(),
        "raw": raw,
    }


def norm_text(s: str) -> str:
    return "".join(str(s or "").split())


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


def is_parrot(reply: str, asr: str) -> bool:
    r = norm_text(reply)
    a = norm_text(asr)
    if not r or not a:
        return False
    if r == a:
        return True
    if len(a) >= 4 and a in r:
        return True
    return cer(a, r) <= 0.15


def entity_fidelity(ref: str, hyp: str) -> Optional[float]:
    """Fraction of ref digits that appear in hyp. None if ref has no digits."""
    needles = _DIGIT_RE.findall(ref or "")
    if not needles:
        return None
    h = hyp or ""
    hit = sum(1 for n in needles if n in h)
    return hit / len(needles)


def _prefix_token_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def assistant_token_weights(
    tokenizer,
    assistant: str,
    *,
    asr_weight: float = 0.2,
    mode_weight: float = 1.0,
    body_weight: float = 1.0,
) -> list[float]:
    """Per-token weights over the assistant string (BPE prefix approximation)."""
    n = _prefix_token_len(tokenizer, assistant)
    if n <= 0:
        return []
    weights = [body_weight] * n

    def _paint(start: int, end: int, w: float) -> None:
        i0 = _prefix_token_len(tokenizer, assistant[:start])
        i1 = _prefix_token_len(tokenizer, assistant[:end])
        for i in range(max(0, i0), min(n, i1)):
            weights[i] = w

    for m in _MODE_RE.finditer(assistant):
        _paint(m.start(), m.end(), mode_weight)
    for m in _ASR_RE.finditer(assistant):
        _paint(m.start(), m.end(), asr_weight)
    return weights


def weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    mask = shift_labels != -100
    if token_weights is None:
        return flat[mask].mean() if mask.any() else flat.sum() * 0.0
    sw = token_weights[:, 1:].to(dtype=flat.dtype)
    num = (flat * sw * mask).sum()
    den = (sw * mask).sum().clamp(min=1e-6)
    return num / den
