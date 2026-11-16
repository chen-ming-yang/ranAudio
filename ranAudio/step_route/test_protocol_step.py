#!/usr/bin/env python3
"""No-GPU parse smoke."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

from ranAudio.protocol import BASE, FC, NOFC, wrap_backend
from ranAudio.step_route.protocol_step import parse_s2_output, strip_nofc_prefix


def main() -> int:
    p = parse_s2_output(f"{FC}{BASE}打开蓝牙")
    assert p["route"] == "fc" and p["transcript"] == "打开蓝牙", p
    p = parse_s2_output(f"{NOFC}你好，我是助手。")
    assert p["route"] == "nofc" and "助手" in p["body"], p
    p = parse_s2_output(f"嗯。{NOFC}今天天气不错")
    assert p["route"] == "nofc" and p["body"] == "今天天气不错", p
    assert wrap_backend("晴") == "<backend>晴</backend>"
    assert strip_nofc_prefix(f"{NOFC}查到了，晴。") == "查到了，晴。"
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
