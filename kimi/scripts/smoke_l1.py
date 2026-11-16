#!/usr/bin/env python3
"""Smoke: fake engine + stable_len helper (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_streaming_engine import FakeStreamingEngine, compute_stable_len  # noqa: E402


class _Tok:
    def encode(self, s: str):
        # 1 char = 1 token for smoke
        return list(range(len(s)))

    def decode(self, ids):
        # reconstruct by length only — tests LCP path with matching text in caller
        return "x" * len(ids)


def test_stable_len():
    tok = _Tok()
    text = "今天天气不错"
    # early chunks → 0
    assert compute_stable_len(
        text,
        tokenizer=tok,
        raw_decoded=text,
        chunk_id=0,
        unfixed_chunk_num=2,
        unfixed_token_num=2,
        is_final=False,
    ) == 0
    # after unfixed: hold back 2 tokens (=2 chars with our tok)
    # decode returns "xx...." same length as end_idx; LCP with Chinese text = 0
    # so use a tokenizer that decodes to prefix of text
    class T2:
        def encode(self, s: str):
            return list(s)

        def decode(self, ids):
            return "".join(ids)

    n = compute_stable_len(
        text,
        tokenizer=T2(),
        raw_decoded=text,
        chunk_id=3,
        unfixed_chunk_num=2,
        unfixed_token_num=2,
        is_final=False,
    )
    assert n == len(text) - 2, n
    assert compute_stable_len(
        text,
        tokenizer=T2(),
        raw_decoded=text,
        chunk_id=3,
        unfixed_chunk_num=2,
        unfixed_token_num=2,
        is_final=True,
    ) == len(text)


def test_fake():
    eng = FakeStreamingEngine()
    eng.load()
    sid = eng.start_session()
    assert sid
    pcm = (np.random.randn(1600).astype(np.float32) * 0.1)
    hyp = None
    for _ in range(12):
        h = eng.feed_pcm16k(pcm)
        if h is not None:
            hyp = h
            assert 0 <= h.stable_len <= len(h.text)
            assert not h.is_final
    final = eng.finish_session()
    assert final.is_final and final.stable_len == len(final.text)
    print("smoke_l1 ok", "last_partial=", hyp.text if hyp else None, "final=", final.text)


if __name__ == "__main__":
    test_stable_len()
    test_fake()
