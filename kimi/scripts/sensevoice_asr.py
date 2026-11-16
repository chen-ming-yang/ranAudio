"""SenseVoice-Small ASR wrapper (FunASR) for the voice-edit demo."""
from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_SENSEVOICE = Path(
    os.environ.get(
        "KIMI_SENSEVOICE",
        "/cmy/cmy/MiniCPM-o-Demo/models/iic/SenseVoiceSmall",
    )
)


class SenseVoiceASR:
    """Thread-safe SenseVoice ASR. Loads once on first use or via load()."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or DEFAULT_SENSEVOICE)
        self.device = device or os.environ.get("KIMI_SENSEVOICE_DEVICE", "cuda:0")
        self._model = None
        self.lock = threading.Lock()
        self.ready = False
        self.load_error: str | None = None

    def load(self) -> None:
        try:
            from funasr import AutoModel

            if not self.model_dir.is_dir():
                raise FileNotFoundError(f"SenseVoice dir missing: {self.model_dir}")
            self._model = AutoModel(
                model=str(self.model_dir),
                trust_remote_code=True,
                device=self.device,
                disable_update=True,
            )
            self.ready = True
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            raise

    def transcribe(self, wav_path: str | Path, *, language: str = "zh") -> str:
        if not self.ready or self._model is None:
            raise RuntimeError(self.load_error or "SenseVoice 未就绪")
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        with self.lock:
            rows = self._model.generate(
                input=str(wav_path),
                language=language,
                use_itn=True,
            )
        raw = ""
        if rows:
            item = rows[0]
            raw = item.get("text", "") if isinstance(item, dict) else str(item)
        return rich_transcription_postprocess(raw or "").strip()
