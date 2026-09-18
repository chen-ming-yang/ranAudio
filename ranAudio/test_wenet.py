#!/usr/bin/env python3
"""Lhotse stream + compact offset index, no 1.5M slurp."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.wenet import (
    IndexedCompact,
    export_compact,
    iter_supervisions,
    load_recordings,
    write_offset_index,
)


def test_stream_and_index() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wav = root / "a.wav"
        wav.write_bytes(b"x")
        rec = root / "recordings.jsonl"
        rec.write_text(
            json.dumps(
                {
                    "id": "R1",
                    "sources": [{"type": "file", "channels": [0], "source": str(wav)}],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        sup = root / "supervisions.jsonl"
        rows = [
            {
                "id": "S1",
                "recording_id": "R1",
                "start": 0.0,
                "duration": 1.0,
                "text": "你好",
                "custom": {"subsets": ["S"]},
            },
            {
                "id": "S2",
                "recording_id": "R1",
                "start": 1.0,
                "duration": 1.0,
                "text": "世界",
                "custom": {"subsets": ["L"]},
            },
        ]
        sup.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        recs = load_recordings(rec)
        got = list(iter_supervisions(sup, recordings=recs, subsets={"S"}, require_audio=True))
        assert len(got) == 1 and got[0].text == "你好"
        out = root / "compact.jsonl"
        n = export_compact(sup, rec, out, subsets={"S"}, require_audio=True)
        assert n == 1
        idx = root / "compact.offsets.npy"
        assert write_offset_index(out, idx) == 1
        indexed = IndexedCompact(out, idx)
        assert len(indexed) == 1
        assert indexed[0].text == "你好"


if __name__ == "__main__":
    test_stream_and_index()
    print("ok")
