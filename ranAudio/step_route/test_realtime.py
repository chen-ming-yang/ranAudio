#!/usr/bin/env python3
"""No-GPU tests for the streaming core: parser / VAD / store / fidelity guard.

Run: python ranAudio/step_route/test_realtime.py
Style follows test_protocol_step.py (main + bare assert, no pytest).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

from ranAudio.step_route.events import Utterance, segment_latency
from ranAudio.step_route.session_store import SessionStore
from ranAudio.step_route.stream_protocol import (
    BASE,
    FC,
    NOFC,
    IncrementalParser,
    fact_ok,
    guard_speak,
    normalize_tags,
)
from ranAudio.step_route.vad import VADConfig, VADStream, energy_vad, load_wav_mono16k


def test_half_emitted_tags() -> None:
    p = IncrementalParser("r1")
    events = []
    for tok in ("<", "|FC", "|>", "<|BASE", "|>", "打开", "蓝牙"):
        events += p.feed(tok)
    events += p.finish()
    assert any(e.kind == "route_first" and e.route == "fc" for e in events), events
    final = [e for e in events if e.kind == "route_final"]
    assert len(final) == 1 and final[0].text == "打开蓝牙" and final[0].route == "fc", final
    assert segment_latency(events, "route_first") is not None
    assert any(e.kind == "transcript_delta" for e in events), events


def test_nofc_body_delta_and_stop() -> None:
    p = IncrementalParser("r2")
    events = p.feed(f"{NOFC}查到了，北京明天多云转晴，18 到 24 度。") + p.finish()
    final = [e for e in events if e.kind == "route_final"][0]
    assert final.route == "nofc", final
    assert "18 到 24 度" in final.text, final
    assert any(e.kind == "body_delta" for e in events)
    p2 = IncrementalParser("r2b")
    ev2 = p2.feed(f"{FC}没有 BASE 的降级串") + p2.finish()
    assert [e for e in ev2 if e.kind == "route_final"][0].route == "other", ev2


def test_tag_slip_normalization() -> None:
    assert normalize_tags("<FC>打开蓝牙</FC>") == f"{FC}打开蓝牙"
    p = IncrementalParser("r3")
    events = p.feed("<FC>好的，已打开蓝牙。</FC>") + p.finish()
    final = [e for e in events if e.kind == "route_final"][0]
    assert final.route == "other" and "蓝牙" in final.text, final


def test_fact_fidelity_guard() -> None:
    speak = "特斯拉最新股价（演示结果）"
    bad = "特斯拉最新股价是 1234.56 美元。"
    text, meta = guard_speak(speak, bad)
    assert meta["fallback"] is True and "演示结果" in text, (text, meta)
    assert fact_ok(speak, bad) is False
    assert fact_ok("气温 10℃到 23℃", "查到了，气温10℃到23℃。") is True
    good = "查到了，特斯拉最新股价（演示结果）"
    text2, meta2 = guard_speak(speak, good)
    assert meta2["fallback"] is False and text2 == good, (text2, meta2)
    assert BASE.startswith("<|")


def _tone(dur: float, sr: int = 16000, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(dur: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * dur), dtype=np.float32)


def test_vad_endpoints() -> None:
    sr = 16000
    pcm = np.concatenate(
        [_silence(0.3, sr), _tone(1.0, sr), _silence(0.6, sr),
         _tone(0.8, sr), _silence(0.6, sr)]
    )
    spans = energy_vad(pcm, sr, VADConfig())
    assert len(spans) == 2, spans
    assert all((b - a) / sr * 1000.0 >= 300 for a, b in spans), spans


def test_vad_max_utt_and_barge_in() -> None:
    sr = 16000
    cfg = VADConfig(max_utt_ms=800, min_speech_ms=100)
    spans = energy_vad(_tone(2.0, sr), sr, cfg)
    assert len(spans) >= 2, spans
    stream = VADStream(sr=sr, source="test")
    stream.feed(_tone(0.2, sr))
    stream.barge_in()
    assert len(stream.buf) == 0
    out = stream.flush()
    assert out == [] or all(isinstance(u, Utterance) for u in out)


def test_utterance_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        utt = Utterance("r9", _tone(0.35))
        path = utt.save_wav(Path(d))
        pcm, sr = load_wav_mono16k(path)
        assert path.is_file() and sr == 16000 and len(pcm) > 4000, (path, sr, len(pcm))


def test_session_store_cancel_and_arbitration() -> None:
    st = SessionStore("s1")
    st.begin("r1", "打开蓝牙")
    assert st.set_pending("r1", "已打开蓝牙。") is True
    assert st.arbitrate("r1", user_speaking=False) == "speak"
    assert st.arbitrate("r1", user_speaking=True) == "hold"
    st.begin("r2", "明天天气")
    assert st.set_pending("r1", "旧结果") is False
    assert st.arbitrate("r1", user_speaking=False) == "drop"
    assert st.set_pending("r2", "多云转晴") is True
    assert st.consume_pending("r2") == "多云转晴"
    assert st.consume_pending("r2") is None
    assert st.set_pending("r2", "又要播") is True
    assert st.cancel("r2") is True
    assert st.arbitrate("r2", user_speaking=False) == "drop"
    st.append_history("已打开蓝牙。")
    assert st.history(1) == ["已打开蓝牙。"]


def test_recorded_backend_replay() -> None:
    from ranAudio.step_route.fastlm_base import RecordedBackend, stream_text

    backend = RecordedBackend.from_baseline()
    if not backend.route_texts:
        return
    stem = sorted(backend.route_texts)[0]
    events = list(backend.route_stream(f"/tmp/{stem}.wav", "r10"))
    assert any(e.kind == "route_first" for e in events), events
    assert any(e.kind == "route_final" for e in events), events
    speak = backend.speaks.get(stem, "")
    if speak:
        ev2 = list(backend.reinject_stream(speak, f"/tmp/{stem}.wav", "r11"))
        reply = [e for e in ev2 if e.kind == "route_final"][0].text
        text, _meta = guard_speak(speak, reply)
        assert speak in text, (speak, text)
    assert any(e.kind == "route_final" for e in stream_text("<|NOFC|>你好", "r12"))


def test_independent_turns() -> None:
    st = SessionStore(independent_turns=True)
    st.begin("a", "第一句")
    st.begin("b", "第二句")
    assert st.set_pending("a", "结果A") is True
    assert st.set_pending("b", "结果B") is True
    assert st.arbitrate("a", user_speaking=False) == "speak"
    assert st.consume_pending("a") == "结果A"
    assert st.consume_pending("b") == "结果B"


def main() -> int:
    test_half_emitted_tags()
    test_nofc_body_delta_and_stop()
    test_tag_slip_normalization()
    test_fact_fidelity_guard()
    test_vad_endpoints()
    test_vad_max_utt_and_barge_in()
    test_utterance_roundtrip()
    test_session_store_cancel_and_arbitration()
    test_independent_turns()
    test_recorded_backend_replay()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
