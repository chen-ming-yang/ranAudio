#!/usr/bin/env python3
"""ASR smoke test for local Kimi-Audio-7B-Instruct."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=str(ROOT / "models" / "Kimi-Audio-7B-Instruct"),
    )
    ap.add_argument(
        "--wav",
        default=str(ROOT / "test_audios" / "asr_example.wav"),
    )
    ap.add_argument("--detokenizer", action="store_true")
    args = ap.parse_args()

    from kimia_infer.api.kimia import KimiAudio

    model = KimiAudio(model_path=args.model, load_detokenizer=args.detokenizer)
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    messages = [
        {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
        {"role": "user", "message_type": "audio", "content": args.wav},
    ]
    _, text = model.generate(messages, **sampling_params, output_type="text")
    print("ASR:", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
