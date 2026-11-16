"""Step-Audio-2-mini as FastLM: audio in → S2 route tokens out."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .protocol_step import REINJECT_SYSTEM, ROUTE_SYSTEM, build_backend_user_text

DEFAULT_MODEL = Path("/cmy/cmy/Step-Audio/Step-Audio-2-mini")
DEFAULT_CODE = Path("/cmy/cmy/Step-Audio/Step-Audio2")


def route_messages(wav: str | Path) -> list[dict[str, Any]]:
    """Route forward: one utterance of audio, no backend span."""
    return [
        {"role": "system", "content": ROUTE_SYSTEM},
        {"role": "human", "content": [{"type": "audio", "audio": str(wav)}]},
        {"role": "assistant", "content": None},
    ]


def reinject_messages(
    speak: str, *, wav: Optional[str | Path] = None
) -> list[dict[str, Any]]:
    """Reinject forward: `<backend>` text plus optional conditioning audio."""
    backend = build_backend_user_text(speak)
    content: list[dict[str, Any]] | str
    if wav and os.path.isfile(str(wav)):
        content = [
            {"type": "text", "text": backend + "\n"},
            {"type": "audio", "audio": str(wav)},
        ]
    else:
        content = backend
    return [
        {"role": "system", "content": REINJECT_SYSTEM},
        {"role": "human", "content": content},
        {"role": "assistant", "content": None},
    ]


def prepare_inputs(model: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble `generate_inputs` exactly like StepAudio2.__call__ L15-44.

    Kept here (workspace) so the streaming backend can call `.llm.generate`
    directly with a streamer, without editing the external Step-Audio repo.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    results, mels = model.apply_chat_template(messages)
    prompt_ids = []
    for item in results:
        if isinstance(item, str):
            prompt_ids.append(
                model.llm_tokenizer(text=item, return_tensors="pt", padding=True)["input_ids"]
            )
        elif isinstance(item, list):
            prompt_ids.append(torch.tensor([item], dtype=torch.int32))
    if len({t.dtype for t in prompt_ids}) > 1:
        prompt_ids = [t.to(torch.long) for t in prompt_ids]
    input_ids = torch.cat(prompt_ids, dim=-1).cuda()
    attention_mask = torch.ones_like(input_ids)
    if mels:
        feats_lengths = torch.tensor([m.size(1) - 2 for m in mels], dtype=torch.int32)
        padded = pad_sequence([m.t() for m in mels], batch_first=True, padding_value=0)
        wavs = padded.transpose(1, 2).cuda()
        wav_lens = feats_lengths.cuda()
    else:
        wavs, wav_lens = None, None
    return {
        "input_ids": input_ids,
        "wavs": wavs,
        "wav_lens": wav_lens,
        "attention_mask": attention_mask,
    }


class StepFastLM:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        code_dir: str | Path = DEFAULT_CODE,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> None:
        self.model_path = Path(model_path)
        self.code_dir = Path(code_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._model = None

    def load(self) -> None:
        code = str(self.code_dir)
        if code not in sys.path:
            sys.path.insert(0, code)
        from stepaudio2 import StepAudio2  # type: ignore

        t0 = time.time()
        self._model = StepAudio2(str(self.model_path))
        print(f"[fastlm] loaded {self.model_path} in {time.time() - t0:.1f}s", flush=True)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("StepFastLM.load() not called")
        return self._model

    def _generate(self, messages: list[dict[str, Any]]) -> tuple[str, float]:
        t0 = time.time()
        _, text, _ = self.model(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            repetition_penalty=1.05,
            top_p=0.9,
        )
        return (text or "").strip(), time.time() - t0

    def route(self, wav: str | Path) -> tuple[str, float]:
        """Audio → raw S2 string (`<|NOFC|>…` or `<|FC|><|BASE|>…`)."""
        wav = str(wav)
        if not os.path.isfile(wav):
            raise FileNotFoundError(wav)
        return self._generate(route_messages(wav))

    def reinject(
        self,
        speak: str,
        *,
        wav: Optional[str | Path] = None,
    ) -> tuple[str, float]:
        """`<backend>speak</backend>` (+ optional audio) → `<|NOFC|>` 拼接稿."""
        if not build_backend_user_text(speak):
            return "", 0.0
        return self._generate(reinject_messages(speak, wav=wav))
