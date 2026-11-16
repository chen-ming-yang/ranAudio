"""S2 route literals for Step-Audio FastLM (reuse ranAudio.protocol constants)."""

from __future__ import annotations

from typing import Any, Optional

from ranAudio.protocol import (
    BACKEND_END,
    BACKEND_START,
    BASE,
    FC,
    NOFC,
    S2_SYSTEM,
    wrap_backend,
)

# Few-shot glued onto S2_SYSTEM so an untrained Step still emits the tags as text.
ROUTE_EXAMPLES = (
    "\n硬性规则：\n"
    f"- 第一个字符起必须是 {FC} 或 {NOFC}（含两侧竖线），禁止输出 <FC> / <NOFC> 这种缺竖线写法。\n"
    f"- {FC} 后必须紧跟 {BASE}，再跟**用户原话转写**（他说了什么就写什么）。\n"
    "- 转写禁止助手口吻：不要写「好的」「正在」「已为你」「我来帮你」等。\n"
    f"- {NOFC} 后直接回答，禁止先复述用户的话。\n"
    "\n示例（只学格式，不要照抄内容）：\n"
    f"闲聊「你是谁」→ {NOFC}我是语音助手。\n"
    f"命令「打开蓝牙」→ {FC}{BASE}打开蓝牙\n"
    f"查信息「帮我搜一下特斯拉最新股价」→ {FC}{BASE}帮我搜一下特斯拉最新股价\n"
    f"听写「请转写：今天会议取消了」→ {NOFC}今天会议取消了\n"
)

ROUTE_SYSTEM = S2_SYSTEM + ROUTE_EXAMPLES

REINJECT_SYSTEM = (
    "你是语音助手。用户消息里的 "
    f"{BACKEND_START}…{BACKEND_END} "
    "是后台执行结果。\n"
    "事实（数字、专名、单位）必须原样保留，禁止改写成其它数字或其它说法。\n"
    f"请输出一句可直接播报的中文，必须以字面 {NOFC} 开头（两侧竖线不能少）。\n"
    "可以加简短衔接（如「查到了」）。\n"
    f"不要回吐 {BACKEND_START}/{BACKEND_END}，不要输出 {FC}、<FC>、工具调用或 JSON。"
)


def _normalize_route_text(text: str) -> str:
    raw = (text or "").strip()
    raw = raw.replace("\u200b", "").replace("\ufeff", "")
    # Common Step slip: drop the pipes around FC/NOFC/BASE.
    raw = raw.replace("<FC>", FC).replace("</FC>", "")
    raw = raw.replace("<NOFC>", NOFC).replace("</NOFC>", "")
    raw = raw.replace("<BASE>", BASE).replace("</BASE>", "")
    return raw


def _cut_body(body: str) -> str:
    body = (body or "").strip()
    # `<backend>` is a user-side INPUT marker. Treating it as a stop token here
    # silently truncated legitimate replies that quoted backend facts.
    for stop in (NOFC, FC, BASE, "<tool_call>", "<|EOT|>", "</FC>", "</NOFC>"):
        i = body.find(stop)
        if i >= 0:
            body = body[:i].strip()
    return body


def parse_s2_output(text: str) -> dict[str, Any]:
    """Parse FastLM raw string into route / body.

    Returns keys: route ('nofc'|'fc'|'other'), body, raw, transcript (fc only).
    """
    orig = (text or "").strip()
    raw = _normalize_route_text(orig)
    if raw.startswith(f"{FC}{BASE}"):
        body = _cut_body(raw[len(FC) + len(BASE) :])
        return {
            "route": "fc",
            "body": body,
            "transcript": body,
            "raw": orig,
        }
    if raw.startswith(NOFC):
        body = _cut_body(raw[len(NOFC) :])
        return {
            "route": "nofc",
            "body": body,
            "transcript": None,
            "raw": orig,
        }
    # Reinject slip: <|FC|> without BASE — keep body for TTS strip, mark other.
    if raw.startswith(FC):
        body = _cut_body(raw[len(FC) :])
        return {
            "route": "other",
            "body": body,
            "transcript": None,
            "raw": orig,
        }
    # Soft recover: tags appear later in the string.
    i_fc = raw.find(f"{FC}{BASE}")
    i_nofc = raw.find(NOFC)
    if i_fc >= 0 and (i_nofc < 0 or i_fc < i_nofc):
        sub = parse_s2_output(raw[i_fc:])
        sub["raw"] = orig
        return sub
    if i_nofc >= 0:
        sub = parse_s2_output(raw[i_nofc:])
        sub["raw"] = orig
        return sub
    return {
        "route": "other",
        "body": _cut_body(raw),
        "transcript": None,
        "raw": orig,
    }


def strip_nofc_prefix(text: str) -> str:
    """Body for TTS: strip route tags; prefer parsed body."""
    p = parse_s2_output(text)
    if p["body"]:
        return p["body"]
    return _cut_body(_normalize_route_text(text))


def build_backend_user_text(speak: str) -> str:
    """User-side reinject payload (text Cond). Empty speak → empty."""
    return wrap_backend(speak)


def parse_backend_span(text: str) -> Optional[str]:
    raw = text or ""
    i0 = raw.find(BACKEND_START)
    i1 = raw.find(BACKEND_END)
    if i0 < 0 or i1 < 0 or i1 < i0:
        return None
    return raw[i0 + len(BACKEND_START) : i1].strip()


def verify_backend_fact(speak: str, reply: str) -> tuple[str, dict[str, Any]]:
    """L2 fidelity guard for the single TTS outlet.

    Delegates to stream_protocol.guard_speak: backend facts must appear
    verbatim, otherwise a structured splice is returned instead.
    """
    from .stream_protocol import guard_speak

    return guard_speak(speak, reply)
