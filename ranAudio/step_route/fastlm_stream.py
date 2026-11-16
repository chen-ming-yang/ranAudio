"""Step-Audio-2-mini streaming decode backend (workspace-only).

`StepAudio2.__call__` runs a whole `generate` and returns a finished string, so
it cannot emit the first route token early. This backend reuses the exposed
`.llm` / `.llm_tokenizer` / `.apply_chat_template`, assembles the same
`generate_inputs` as `stepaudio2.py:14-44` via `fastlm.prepare_inputs`, and
runs `generate` on a worker thread with a `TextIteratorStreamer`.

No file under /cmy/cmy/Step-Audio is modified. If transformers cannot be
imported (e.g. the venv's hub version conflict), it falls back to whole-string
generate replayed as deltas, so the event contract is preserved.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from .events import Clock, StreamEvent
from .fastlm import (
    DEFAULT_CODE,
    DEFAULT_MODEL,
    prepare_inputs,
    reinject_messages,
    route_messages,
)
from .stream_protocol import IncrementalParser


def _clean_delta(text: str) -> str:
    """Drop unprintable audio-token decodes; keep CJK/ASCII punctuation."""
    return "".join(ch for ch in (text or "") if ch.isprintable() or ch in "\n\r\t")


class StepStreamBackend:
    name = "step-stream"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        code_dir: str | Path = DEFAULT_CODE,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        fallback_chunk_chars: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.code_dir = Path(code_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.fallback_chunk_chars = int(fallback_chunk_chars)
        self._model = None
        self.streamer_available = False

    def load(self) -> None:
        code = str(self.code_dir)
        if code not in sys.path:
            sys.path.insert(0, code)
        from stepaudio2 import StepAudio2  # type: ignore

        self._model = StepAudio2(str(self.model_path))
        try:
            from transformers import TextIteratorStreamer  # noqa: F401

            self.streamer_available = True
        except Exception as exc:
            print(f"[step-stream] streamer unavailable ({exc}); whole-string fallback", flush=True)
        print(f"[step-stream] loaded {self.model_path}", flush=True)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("StepStreamBackend.load() not called")
        return self._model

    def _gen_kwargs(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
            "repetition_penalty": 1.05,
            "top_p": 0.9,
        }

    def _deltas(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        if not self.streamer_available:
            _, text, _ = self.model(messages, **self._gen_kwargs())
            text = (text or "").strip()
            for i in range(0, len(text), self.fallback_chunk_chars):
                yield text[i : i + self.fallback_chunk_chars]
            return

        from transformers import GenerationConfig, TextIteratorStreamer

        inputs = prepare_inputs(self.model, messages)
        streamer = TextIteratorStreamer(
            self.model.llm_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        cfg = GenerationConfig(
            pad_token_id=self.model.llm_tokenizer.pad_token_id,
            eos_token_id=self.model.eos_token_id,
            **self._gen_kwargs(),
        )
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                self.model.llm.generate(
                    **inputs,
                    generation_config=cfg,
                    streamer=streamer,
                    tokenizer=self.model.llm_tokenizer,
                )
            except BaseException as exc:  # keep the consumer unblocked
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=_run, name="step-stream-generate", daemon=True)
        worker.start()
        yield from streamer
        worker.join(timeout=2.0)
        if errors:
            raise RuntimeError(f"Step generate failed: {errors[0]}")

    def _events(self, messages: list[dict[str, Any]], request_id: str) -> Iterator[StreamEvent]:
        parser = IncrementalParser(request_id, Clock())
        for delta in self._deltas(messages):
            yield from parser.feed(_clean_delta(delta))
        yield from parser.finish()

    def route_stream(self, wav: str | Path, request_id: str) -> Iterator[StreamEvent]:
        wav = str(wav)
        if not Path(wav).is_file():
            raise FileNotFoundError(wav)
        yield from self._events(route_messages(wav), request_id)

    def reinject_stream(
        self, speak: str, wav: Optional[str | Path], request_id: str
    ) -> Iterator[StreamEvent]:
        if not (speak or "").strip():
            return
        yield from self._events(reinject_messages(speak, wav=wav), request_id)
