#!/usr/bin/env python3
"""MiniCPM5 S2 first-token gate + prefix-chunk route probe.

The MiniCPM5 stack uses transformers 5.15.0.dev0 while the Step/SlowLM side
uses 4.49.0 / another source tree, so this backend is split in two:

  * `--serve` worker  : runs under minicpm5 python, owns the GPU model.
  * MiniCPMProbeBackend: lives in the coordinator process (Step venv), talks
                         JSON-lines over stdin/stdout. No cross-import.

The S2 tokenizer already has the six special tokens (CHAT_SPECIAL_TOKENS), so
the very first generated token IS `<|FC|>` / `<|NOFC|>`: a real token-level
gate, unlike Step's plain-text tags.

Prefix probe: re-encode cumulative audio prefixes and generate only 1 token per
chunk, approximating "route while listening". This is O(k^2) re-encoding, NOT
chunk-causal streaming (S1 retraining is still required for 80-160 ms).

One-shot diagnostic (run directly under minicpm5 python):
  python -m ranAudio.step_route.fastlm_minicpm --wav /path/a.wav --out /tmp/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any, Iterator, Optional

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
import ranAudio._boot  # noqa: F401

from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, SENSEVOICE

from .events import Clock, StreamEvent
from .fastlm_base import stream_text
from .stream_protocol import IncrementalParser, NOFC

MINICPM5_PYTHON = Path("/cmy/cmy/miniconda3/envs/minicpm5/bin/python")
ADAPTER = RUNS_DIR / "s2_fc_nofc_full" / "mix_llmcont"
S15_ENC = RUNS_DIR / "s15_asr" / "b_ms_llm_cont" / "encoder_best.pt"
S15_PROJ = RUNS_DIR / "s15_asr" / "b_ms_llm_cont" / "projector_best.pt"


# --------------------------------------------------------------------------- #
# Worker side (minicpm5 interpreter only)
# --------------------------------------------------------------------------- #
class _MiniCPMWorker:
    def __init__(self, adapter: Path, encoder: Path, projector: Path) -> None:
        self.adapter = adapter
        self.encoder = encoder
        self.projector = projector
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from ranAudio.model import MiniCPM5Audio
        from ranAudio.protocol import S2_SYSTEM

        self.system = S2_SYSTEM
        tok_dir = self.adapter / "tokenizer"
        tokenizer = (
            AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
            if tok_dir.is_dir()
            else None
        )
        model = MiniCPM5Audio.from_pretrained(
            MINICPM5,
            MINICPMO,
            projector_path=self.projector,
            encoder_path=self.encoder,
            encoder_backend="sensevoice",
            sensevoice_path=SENSEVOICE,
            tokenizer=tokenizer,
        )
        llm_ckpt = self.adapter / "llm_final.safetensors"
        if not llm_ckpt.is_file():
            raise FileNotFoundError(llm_ckpt)
        # Load on CPU then copy param-by-param: staging the 4.7 GB state dict on
        # GPU as well would OOM when the GPU is already near capacity.
        sd = load_file(str(llm_ckpt), device="cpu")
        model.llm.load_state_dict(sd)
        del sd
        model.eval()
        self.model = model
        self.tokenizer = model.tokenizer
        print(f"[minicpm-probe] loaded {llm_ckpt}", flush=True)

    def _first_token(self, audio) -> tuple[str, float, str]:
        """One-token gate on a float32 mono-16k array."""
        import numpy as np
        import torch

        t0 = time.perf_counter()
        embeds = self.model.encode_wav(np.asarray(audio, dtype=np.float32))
        n = int(embeds.shape[0])
        prefix = self.model._apply_chat(
            n_audio=n,
            tools=None,
            assistant=None,
            add_generation_prompt=True,
            system=self.system,
        )
        input_ids = self.tokenizer(
            prefix, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device)
        inputs_embeds = self.model._embed_ids(input_ids, embeds)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = (
            out[0][input_ids.shape[1] :]
            if out.shape[1] > input_ids.shape[1]
            else out[0]
        )
        tok = self.tokenizer.decode(new_ids, skip_special_tokens=False).strip()
        from ranAudio.protocol import FC

        route = "fc" if tok.startswith(FC) else "nofc" if tok.startswith(NOFC) else "other"
        return route, (time.perf_counter() - t0) * 1000.0, tok

    def route(self, wav: str) -> dict[str, Any]:
        import librosa

        from ranAudio.paths import resolve_wav

        t0 = time.perf_counter()
        audio, _ = librosa.load(str(resolve_wav(wav)), sr=16000, mono=True)
        route, first_ms, tok = self._first_token(audio)
        full_text = self.model.generate_text(
            audio, tools=None, system=self.system, max_new_tokens=128
        )
        return {
            "route": route,
            "first_token": tok,
            "first_token_ms": round(first_ms, 3),
            "full_text": full_text,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def probe(self, wav: str, chunk_ms: int, max_chunks: int) -> dict[str, Any]:
        import librosa

        from ranAudio.paths import resolve_wav

        t0 = time.perf_counter()
        audio, _ = librosa.load(str(resolve_wav(wav)), sr=16000, mono=True)
        step = max(1, int(16000 * chunk_ms / 1000))
        n_chunks = max(1, min(int(max_chunks), (len(audio) + step - 1) // step))
        chunks: list[dict[str, Any]] = []
        streak = 0
        last_route = ""
        for k in range(1, n_chunks + 1):
            prefix = audio[: min(len(audio), k * step)]
            t1 = time.perf_counter()
            route, _ms, tok = self._first_token(prefix)
            chunks.append(
                {
                    "chunk": k,
                    "audio_ms": round(len(prefix) / 16.0, 1),
                    "route": route,
                    "token": tok,
                    "t_ms": round((time.perf_counter() - t1) * 1000.0, 3),
                    "since_start_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                }
            )
            streak = streak + 1 if route == last_route == "fc" else 0
            last_route = route
            if streak >= 1:  # two consecutive consistent FC chunks -> early stop
                break
        return {
            "chunks": chunks,
            "total_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "complexity": "O(k^2) prefix re-encode; not chunk-causal",
        }

    @staticmethod
    def reinject(speak: str) -> dict[str, Any]:
        # MiniCPM5 S2 was never trained on the backend splicing bucket, so a
        # generative copy would hallucinate. Structured splice is deterministic.
        text = f"{NOFC}{speak}".strip()
        return {"text": text, "structured_fallback": True}


def _serve(adapter: Path, encoder: Path, projector: Path) -> None:
    worker = _MiniCPMWorker(adapter, encoder, projector)
    try:
        worker.load()
        print(json.dumps({"event": "ready"}), flush=True)
    except Exception as exc:
        print(json.dumps({"event": "error", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("cmd") == "quit":
            break
        rid = req.get("id") or ""
        try:
            if req.get("cmd") == "route":
                payload: dict[str, Any] = worker.route(str(req.get("wav") or ""))
            elif req.get("cmd") == "probe":
                payload = worker.probe(
                    str(req.get("wav") or ""),
                    int(req.get("chunk_ms") or 250),
                    int(req.get("max_chunks") or 8),
                )
            elif req.get("cmd") == "reinject":
                payload = _MiniCPMWorker.reinject(str(req.get("speak") or ""))
            else:
                continue
            print(json.dumps({"id": rid, "event": "result", **payload}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps({"id": rid, "event": "result", "error": f"{type(exc).__name__}: {exc}"},
                           ensure_ascii=False),
                flush=True,
            )


# --------------------------------------------------------------------------- #
# Coordinator side (Step venv / any interpreter, no MiniCPM import)
# --------------------------------------------------------------------------- #
class MiniCPMProbeBackend:
    name = "minicpm-probe"

    def __init__(
        self,
        *,
        python: Path = MINICPM5_PYTHON,
        adapter: Path = ADAPTER,
        encoder: Path = S15_ENC,
        projector: Path = S15_PROJ,
        chunk_ms: int = 250,
        max_chunks: int = 8,
    ) -> None:
        self.python = Path(python)
        self.adapter = Path(adapter)
        self.encoder = Path(encoder)
        self.projector = Path(projector)
        self.chunk_ms = int(chunk_ms)
        self.max_chunks = int(max_chunks)
        self.proc: Optional[Popen] = None
        self.load_error = ""
        self._ready = threading.Event()
        self._cv = threading.Condition()
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._stderr_tail: list[str] = []

    def start(self, timeout: float = 300.0) -> None:
        if not self.python.is_file():
            raise FileNotFoundError(f"minicpm5 python not found: {self.python}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_PROJECT)
        env.pop("PYTHONNOUSERSITE", None)
        cmd = [
            str(self.python), "-u", "-m", "ranAudio.step_route.fastlm_minicpm", "--serve",
            "--adapter", str(self.adapter),
            "--encoder", str(self.encoder),
            "--projector", str(self.projector),
        ]
        self.proc = Popen(
            cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,
            env=env, text=True, bufsize=1, cwd=str(_PROJECT),
        )
        threading.Thread(target=self._read_stdout, name="minicpm-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="minicpm-stderr", daemon=True).start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError(f"MiniCPM worker not ready in {timeout}s: {self.load_error}")
        if self.load_error:
            err = self.load_error
            self.close()
            raise RuntimeError(err)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    def _request(self, cmd: str, timeout: float = 180.0, **payload) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("MiniCPM worker not running")
        rid = uuid.uuid4().hex
        with self._cv:
            self._pending[rid] = []
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"cmd": cmd, "id": rid, **payload}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        with self._cv:
            while not self._pending[rid] and time.time() < deadline:
                self._cv.wait(timeout=1.0)
            box = self._pending.pop(rid, [])
        if not box:
            raise TimeoutError(f"MiniCPM {cmd} timeout")
        item = box[0]
        if item.get("error"):
            raise RuntimeError(item["error"])
        return item

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if msg.get("event") == "ready":
                self._ready.set()
                continue
            if msg.get("event") == "error":
                self.load_error = str(msg.get("error"))
                self._ready.set()
                continue
            rid = msg.get("id")
            if rid:
                with self._cv:
                    if rid in self._pending:
                        self._pending[rid].append(msg)
                        self._cv.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 50:
                self._stderr_tail = self._stderr_tail[-50:]
            print(f"[minicpm:err] {line.rstrip()}", flush=True)

    def route_stream(self, wav: str | Path, request_id: str) -> Iterator[StreamEvent]:
        resp = self._request("route", wav=str(wav))
        yield StreamEvent(
            "route_first", request_id, float(resp.get("first_token_ms") or 0.0),
            route=resp.get("route"), meta={"backend": self.name, "token": resp.get("first_token")},
        )
        clock = Clock()
        parser = IncrementalParser(request_id, clock)
        events = parser.feed(resp.get("full_text") or "") + parser.finish()
        for ev in events:
            if ev.kind == "route_first":
                continue
            yield ev

    def probe_stream(self, wav: str | Path, request_id: str) -> Iterator[StreamEvent]:
        resp = self._request(
            "probe", wav=str(wav), chunk_ms=self.chunk_ms, max_chunks=self.max_chunks
        )
        for c in resp.get("chunks", []):
            yield StreamEvent(
                "route_first", request_id, float(c.get("since_start_ms") or 0.0),
                route=c.get("route"),
                meta={"backend": self.name, "chunk": c.get("chunk"),
                      "audio_ms": c.get("audio_ms"), "probe_t_ms": c.get("t_ms")},
            )

    def reinject_stream(
        self, speak: str, wav: Optional[str | Path], request_id: str
    ) -> Iterator[StreamEvent]:
        if not (speak or "").strip():
            return
        resp = self._request("reinject", speak=speak)
        yield from stream_text(resp.get("text") or "", request_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--encoder", default=str(S15_ENC))
    ap.add_argument("--projector", default=str(S15_PROJ))
    ap.add_argument("--wav", action="append", default=[], help="repeatable one-shot diagnostic")
    ap.add_argument("--probe", action="store_true", help="prefix probe instead of route")
    ap.add_argument("--chunk-ms", type=int, default=250)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.serve:
        _serve(Path(args.adapter), Path(args.encoder), Path(args.projector))
        return 0
    if not args.wav:
        ap.error("--wav required unless --serve")
    worker = _MiniCPMWorker(Path(args.adapter), Path(args.encoder), Path(args.projector))
    worker.load()
    results: list[dict[str, Any]] = []
    for i, wav in enumerate(args.wav):
        res = (
            worker.probe(wav, args.chunk_ms, args.max_chunks)
            if args.probe
            else worker.route(wav)
        )
        res["wav"] = wav
        res["warmup"] = i == 0
        results.append(res)
        print(f"[minicpm-probe] {i + 1}/{len(args.wav)} {Path(wav).name}: {res.get('route') or 'probe'}", flush=True)
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
