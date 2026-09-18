"""Judge JSON + legacy MiniCPM5 tool-call XML/JSON parsers."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_REPO_SYS = (
    "你是车机语音助手。根据用户语音调用提供的工具。"
    "有参工具必须在 arguments 里填写听到的地点、温度或音量；没听到的键不要编造。"
    "只输出 function call，不要寒暄，不要解释。"
)

_JUDGE_SYS = (
    "你是车机语音助手。根据用户语音输出一行 JSON。"
    "字段：asr（转写字符串）、need_fc（布尔）。"
    "车机控制意图（蓝牙、导航、空调、音量、车窗、电话、音乐、座椅加热、大灯等）"
    "need_fc 为 true，此时只输出 asr 与 need_fc，不要填 reply，不要编造工具名或参数。"
    "闲聊、问答、与车机工具无关时 need_fc 为 false，并额外填写 reply（你作为普通助手的中文回复）。"
    "不要输出 function call / XML / 工具调用。"
)

# MiniCPM5 chat_template.jinja:
# <function name="fn"><param name="k">v</param></function>
_FN5 = re.compile(r'<function name="([^"]+)">(.*?)</function>', re.DOTALL)
_PARAM5 = re.compile(r'<param name="([^"]+)">(.*?)</param>', re.DOTALL)
_TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FN_QWEN = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAM_QWEN = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def format_tool_xml(name: str, arguments: Optional[dict[str, Any]] = None) -> str:
    args = arguments or {}
    body = "".join(f'<param name="{k}">{_dump_param(v)}</param>' for k, v in args.items())
    return f'<function name="{name}">{body}</function>'


def format_judge_json(
    asr: str,
    need_fc: bool,
    reply: Optional[str] = None,
) -> str:
    """need_fc=true: only asr+need_fc. need_fc=false: also include reply (2B as LLM)."""
    obj: dict[str, Any] = {"asr": asr, "need_fc": bool(need_fc)}
    if not need_fc:
        obj["reply"] = str(reply or "")
    return json.dumps(obj, ensure_ascii=False)


def parse_judge(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    obj: Any = None
    start = raw.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        m = _JSON_OBJ.search(raw)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(obj, dict) or "asr" not in obj or "need_fc" not in obj:
        return None
    need = obj["need_fc"]
    if isinstance(need, str):
        need = need.strip().lower() in ("1", "true", "yes", "y")
    else:
        need = bool(need)
    out: dict[str, Any] = {"asr": str(obj.get("asr") or ""), "need_fc": need}
    if "reply" in obj and obj["reply"] is not None:
        out["reply"] = str(obj["reply"])
    elif not need:
        out["reply"] = ""
    return out


def _dump_param(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    out: list[dict[str, Any]] = []
    for fn in _FN5.finditer(text):
        arguments: dict[str, Any] = {}
        for pm in _PARAM5.finditer(fn.group(2)):
            arguments[pm.group(1)] = _parse_value(pm.group(2))
        out.append({"name": fn.group(1), "arguments": arguments})
    if out:
        return out
    for block in _TOOL_BLOCK.finditer(text):
        inner = block.group(1).strip()
        fn = _FN_QWEN.search(inner)
        if fn:
            arguments = {}
            for pm in _PARAM_QWEN.finditer(fn.group(2)):
                arguments[pm.group(1)] = _parse_value(pm.group(2))
            out.append({"name": fn.group(1), "arguments": arguments})
            continue
        m = _JSON_OBJ.search(inner)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                out.append({"name": str(obj["name"]), "arguments": args})
    return out


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def fc_system_prompt() -> str:
    return _REPO_SYS


def judge_system_prompt() -> str:
    return _JUDGE_SYS
