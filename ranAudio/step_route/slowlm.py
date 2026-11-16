#!/usr/bin/env python3
"""Qwen3.5-2B SlowLM: transcript in → speak text out.

Runs under qwen3-compare + transformers-main (same stack as glyph_postprocess).
FastLM process (Step venv) must not import this model's transformers;
use SlowLMClient which spawns `--serve`.
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
from typing import Optional

DEFAULT_MODEL = Path("/cmy/cmy/Qwen/Qwen3.5-2B")
DEFAULT_PYTHON = Path("/cmy/cmy/miniconda3/envs/qwen3-compare/bin/python")
TRANSFORMERS_SRC = Path("/cmy/cmy/Qwen/transformers-main/src")

SLOW_SYSTEM = (
    "你是车机/语音助手的慢路径执行模型。输入是用户原话转写。\n"
    "完成需要的推理或工具规划后，只输出一句给用户听的中文播报稿（speak）。\n"
    "不要输出 JSON、XML、tool_call、路由标记。不要复述转写原文当答案。\n"
    "若是控车类指令且无真实执行环境，用简短确认（如「好的，已打开蓝牙。」）。\n"
    "若是查天气/股价/搜索等，用合理占位事实作答，并标明是演示结果。"
)


def _load_model(model_path: Path):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception:
        processor = None
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, processor


def _generate_speak(
    model,
    tokenizer,
    transcript: str,
    *,
    max_new_tokens: int = 256,
) -> str:
    import torch

    user = f"【转写】{(transcript or '').strip()}\n【播报】"
    messages = [
        {"role": "system", "content": SLOW_SYSTEM},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = SLOW_SYSTEM + "\n" + user
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            or tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    # Drop accidental prefixes.
    for p in ("【播报】", "播报：", "播报:"):
        if text.startswith(p):
            text = text[len(p) :].strip()
    return text


class SlowLMClient:
    """Long-lived SlowLM worker in a separate interpreter."""

    def __init__(
        self,
        *,
        python: Path = DEFAULT_PYTHON,
        model: Path = DEFAULT_MODEL,
        transformers_src: Path = TRANSFORMERS_SRC,
        max_new_tokens: int = 256,
    ) -> None:
        self.python = Path(python)
        self.model = Path(model)
        self.transformers_src = Path(transformers_src)
        self.max_new_tokens = int(max_new_tokens)
        self.proc: Optional[Popen] = None
        self.load_error = ""
        self._ready = threading.Event()
        self._cv = threading.Condition()
        self._pending: dict[str, list] = {}
        self._cancelled: set[str] = set()
        self._late: list[dict] = []
        self._stderr_tail: list[str] = []

    def start(self, timeout: float = 300.0) -> None:
        if not self.python.is_file():
            raise FileNotFoundError(f"slowlm python not found: {self.python}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.transformers_src)
        env.pop("PYTHONNOUSERSITE", None)
        cmd = [
            str(self.python),
            "-u",
            str(Path(__file__).resolve()),
            "--serve",
            "--model",
            str(self.model),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        self.proc = Popen(
            cmd,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, name="slow-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="slow-stderr", daemon=True).start()
        if not self._ready.wait(timeout):
            self.close()
            raise TimeoutError(
                f"SlowLM worker not ready in {timeout}s: {self.load_error or self._stderr_tail[-5:]}"
            )
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

    def speak(
        self, transcript: str, timeout: float = 120.0, *, request_id: str | None = None
    ) -> str:
        """SlowLM call. `request_id` ties the result back to the L2 FC emit.

        Old callers (`speak(transcript)`) keep working; only the streaming
        coordinator passes an explicit id so cancel / stale-drop can work.
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("SlowLM worker not running")
        rid = request_id or uuid.uuid4().hex
        with self._cv:
            self._pending[rid] = []
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps(
                {"cmd": "speak", "id": rid, "request_id": rid, "transcript": transcript},
                ensure_ascii=False,
            )
            + "\n"
        )
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        with self._cv:
            while not self._pending[rid] and time.time() < deadline and rid not in self._cancelled:
                self._cv.wait(timeout=1.0)
            box = self._pending.pop(rid, [])
        if not box:
            if rid in self._cancelled:
                raise RuntimeError(f"SlowLM request cancelled: {rid}")
            raise TimeoutError(f"SlowLM speak timeout: {transcript[:40]}")
        item = box[0]
        if item.get("error"):
            raise RuntimeError(item["error"])
        return str(item.get("speak") or "").strip()

    def cancel(self, request_id: str) -> None:
        """barge_in / new request: reject the id; late results are recorded, not dropped silently."""
        with self._cv:
            self._cancelled.add(request_id)
            self._pending.pop(request_id, None)
            self._cv.notify_all()
        if self.proc and self.proc.stdin and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "cancel", "id": request_id}) + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") == "ready":
                self._ready.set()
                continue
            if msg.get("event") == "error":
                self.load_error = str(msg.get("error") or "unknown")
                self._ready.set()
                continue
            rid = msg.get("id")
            if rid:
                with self._cv:
                    if rid in self._cancelled:
                        self._late.append({"id": rid, "reason": "cancelled", "msg": msg})
                    elif rid in self._pending:
                        self._pending[rid].append(msg)
                    else:
                        self._late.append({"id": rid, "reason": "late_or_unknown", "msg": msg})
                    self._cv.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 50:
                self._stderr_tail = self._stderr_tail[-50:]
            print(f"[slowlm:err] {line.rstrip()}", flush=True)


def _serve(model_path: Path, max_new_tokens: int) -> None:
    try:
        model, tokenizer, _ = _load_model(model_path)
        print(json.dumps({"event": "ready"}), flush=True)
    except Exception as e:
        print(json.dumps({"event": "error", "error": f"{type(e).__name__}: {e}"}), flush=True)
        return
    cancelled: set[str] = set()
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
        if req.get("cmd") == "cancel":
            cancelled.add(str(req.get("id") or ""))
            continue
        if req.get("cmd") != "speak":
            continue
        rid = req.get("id") or ""
        if rid in cancelled:
            print(
                json.dumps({"id": rid, "request_id": rid, "error": "cancelled"}, ensure_ascii=False),
                flush=True,
            )
            continue
        try:
            speak = _generate_speak(
                model,
                tokenizer,
                str(req.get("transcript") or ""),
                max_new_tokens=max_new_tokens,
            )
            print(
                json.dumps({"id": rid, "request_id": rid, "speak": speak}, ensure_ascii=False),
                flush=True,
            )
        except Exception as e:
            print(
                json.dumps(
                    {"id": rid, "request_id": rid, "error": f"{type(e).__name__}: {e}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-2B SlowLM worker / one-shot")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--transcript", default="")
    args = ap.parse_args()
    # Ensure transformers-main when launched directly.
    src = str(TRANSFORMERS_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    if args.serve:
        _serve(Path(args.model), args.max_new_tokens)
        return 0
    if not args.transcript:
        ap.error("--transcript required unless --serve")
    model, tokenizer, _ = _load_model(Path(args.model))
    print(_generate_speak(model, tokenizer, args.transcript, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
