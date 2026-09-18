#!/usr/bin/env python3
"""Quick unit checks for judge JSON parse/format."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.tools import format_judge_json, parse_judge


def test_roundtrip() -> None:
    s_fc = format_judge_json("导航去家", True)
    assert s_fc == '{"asr": "导航去家", "need_fc": true}'
    assert "reply" not in s_fc
    assert parse_judge(s_fc) == {"asr": "导航去家", "need_fc": True}

    s_chat = format_judge_json("今天天气怎么样", False, reply="建议看手机天气。")
    assert '"need_fc": false' in s_chat and '"reply":' in s_chat
    p = parse_judge(s_chat)
    assert p == {
        "asr": "今天天气怎么样",
        "need_fc": False,
        "reply": "建议看手机天气。",
    }
    assert parse_judge('前缀 {"asr":"你好","need_fc":false,"reply":"嗨"} 后缀') == {
        "asr": "你好",
        "need_fc": False,
        "reply": "嗨",
    }
    assert parse_judge("not json") is None


if __name__ == "__main__":
    test_roundtrip()
    print("ok")
