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

from ranAudio.model import AUDIO_END, AUDIO_PAD, AUDIO_START, build_user_content
from ranAudio.protocol import (
    BACKEND_END,
    BACKEND_START,
    BASE,
    FC,
    NOFC,
    S2_SPECIAL_TOKENS,
    cer,
    format_fc_base,
    format_nofc,
    format_route,
    format_speak,
    is_parrot,
    parse_backend,
    parse_stream,
    weighted_lm_loss,
    wrap_backend,
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


def test_s2_backend() -> None:
    assert wrap_backend("") == ""
    assert wrap_backend("  ") == ""
    span = wrap_backend("北京明天多云转晴，18到24度。")
    assert span == f"{BACKEND_START}北京明天多云转晴，18到24度。{BACKEND_END}"
    assert parse_backend(span) == "北京明天多云转晴，18到24度。"
    assert parse_backend("无标签") is None
    assert format_nofc("查到了，北京明天多云转晴，18到24度。") == (
        f"{NOFC}查到了，北京明天多云转晴，18到24度。"
    )
    assert format_fc_base("打开蓝牙") == f"{FC}{BASE}打开蓝牙"
    assert BACKEND_START in S2_SPECIAL_TOKENS and BACKEND_END in S2_SPECIAL_TOKENS


def test_build_user_content() -> None:
    audio3 = AUDIO_START + AUDIO_PAD * 3 + AUDIO_END
    assert build_user_content(n_audio=3) == audio3
    assert build_user_content(n_audio=3, backend="") == audio3
    mixed = build_user_content(n_audio=2, backend="北京明天多云转晴，18到24度。")
    assert mixed.startswith(f"{BACKEND_START}北京明天多云转晴，18到24度。{BACKEND_END}")
    assert mixed.endswith(AUDIO_START + AUDIO_PAD * 2 + AUDIO_END)
    assert build_user_content(user_text="打开蓝牙", n_audio=4, backend="x") == "打开蓝牙"


def test_weighted_loss() -> None:
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[-100, 1, 2, 3]])
    w = torch.tensor([[0.0, 1.0, 0.2, 1.0]])
    loss = weighted_lm_loss(logits, labels, w)
    assert torch.isfinite(loss)


def test_s2_token_weights() -> None:
    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    from ranAudio.protocol import assistant_token_weights

    tok = _Tok()
    fc = format_fc_base("打开蓝牙")
    w = assistant_token_weights(tok, fc, asr_weight=0.2, mode_weight=1.0)
    assert w[: len(FC)] == [1.0] * len(FC)
    assert w[len(FC) : len(FC) + len(BASE)] == [1.0] * len(BASE)
    assert all(x == 0.2 for x in w[len(FC) + len(BASE) :])
    nofc = format_nofc("我是语音助手")
    w2 = assistant_token_weights(tok, nofc, asr_weight=0.2, mode_weight=1.0)
    assert w2[: len(NOFC)] == [1.0] * len(NOFC)
    assert all(x == 1.0 for x in w2[len(NOFC) :])


if __name__ == "__main__":
    test_format_parse()
    test_cer_parrot()
    test_s2_backend()
    test_build_user_content()
    test_weighted_loss()
    test_s2_token_weights()
    print("ok")
