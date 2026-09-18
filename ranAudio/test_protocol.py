#!/usr/bin/env python3
"""Unit checks for stream tags, CER, weighted loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.protocol import (
    cer,
    format_route,
    format_speak,
    is_parrot,
    parse_stream,
    weighted_lm_loss,
)


def test_format_parse() -> None:
    s = format_speak("今天不错")
    p = parse_stream(s)
    assert p["mode"] == "speak" and p["reply"] == "今天不错" and p["asr"] == ""
    r = format_route("打开蓝牙", filler="好的")
    q = parse_stream(r)
    assert q["mode"] == "route" and q["asr"] == "打开蓝牙" and q["filler"] == "好的"
    assert parse_stream("hello")["mode"] is None


def test_cer_parrot() -> None:
    assert cer("打开蓝牙", "打开蓝牙") == 0.0
    assert cer("打开蓝牙", "关闭蓝牙") > 0
    assert is_parrot("打开蓝牙", "打开蓝牙")
    assert not is_parrot("好的，已打开。", "打开蓝牙")


def test_weighted_loss() -> None:
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[-100, 1, 2, 3]])
    w = torch.tensor([[0.0, 1.0, 0.2, 1.0]])
    loss = weighted_lm_loss(logits, labels, w)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_format_parse()
    test_cer_parrot()
    test_weighted_loss()
    print("ok")
