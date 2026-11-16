#!/usr/bin/env python3
"""Glyph-norm rewriter: Qwen3.5-2B + LoRA, one ASR final in, full text out.

The L1 process (vLLM / transformers 4.x) must not import this model's stack.
It spawns this file under qwen3-compare with transformers-main on PYTHONPATH.
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
from subprocess import Popen, PIPE

KIMI = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path("/cmy/cmy/Qwen/Qwen3.5-2B")
DEFAULT_ADAPTER = KIMI / "eval_name_clarify" / "sft_out" / "qwen35_2b_lora_v3"
DEFAULT_PYTHON = Path("/cmy/cmy/miniconda3/envs/qwen3-compare/bin/python")
TRANSFORMERS_SRC = Path("/cmy/cmy/Qwen/transformers-main/src")

INSTRUCTION = (
    "下面是语音识别原文。若句中用组词或偏旁说明字形，"
    "把被说明的字和说明本身改成同一个正确字。"
    "若用「第N条」「第N点」或「下一条」逐条列举，收成从 1. 开始、每条一行的编号。"
    "若在报菜名或罗列一串名称，每个名称单独一行，不加序号。"
    "若用两个部件拼出这个字，把被说明的字改成这个字。"
    "被说明的字认成了别的字时，以说明为准。"
    "若用「第N段」或「下一段」分段，或前后说的是不同的事，分成若干段，段与段之间空一行。"
    "钟点写成「时:分」，算式写成数字和加减乘除符号，句里的数目写成阿拉伯数字。"
    "「第N条」「第N段」仍按列表和分段处理。"
    "没有这类情况则原样输出。只输出全文。\n"
    "【原文】{asr}"
)

BENCH_TEXTS = [
    "我叫陈明阳，明是名字的明，阳是提手旁的阳。",
    "我叫陈明阳，名是名字的名，杨是提手旁的杨。",
    "我是陈名杨，名是名字的名，杨是铁手旁的羊",
    "我是李小明，明是明日的明",
    "今天天气不错，适合出门散步。",
]


class GlyphRewriteClient:
    """Long-lived worker. rewrite() is safe to call from a worker thread."""

    def __init__(
        self,
        *,
        python: Path = DEFAULT_PYTHON,
        base: Path = DEFAULT_BASE,
        adapter: Path = DEFAULT_ADAPTER,
        max_new_tokens: int = 128,
    ) -> None:
        self.python = Path(python)
        self.base = Path(base)
        self.adapter = Path(adapter)
        self.max_new_tokens = int(max_new_tokens)
        self.proc: Popen | None = None
        self.load_error = ""
        self._ready = threading.Event()
        self._cv = threading.Condition()
        self._pending: dict[str, list] = {}
        self._stderr_tail: list[str] = []

    def start(self, timeout: float = 180.0) -> None:
        if not self.python.is_file():
            raise FileNotFoundError(f"llm python not found: {self.python}")
        if not self.adapter.is_dir():
            raise FileNotFoundError(f"adapter not found: {self.adapter}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(TRANSFORMERS_SRC)
        env.pop("PYTHONNOUSERSITE", None)
        cmd = [
            str(self.python),
            "-u",
            str(Path(__file__).resolve()),
            "--serve",
            "--base",
            str(self.base),
            "--adapter",
            str(self.adapter),
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
        threading.Thread(target=self._read_stdout, name="glyph-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="glyph-stderr", daemon=True).start()
        if not self._ready.wait(timeout):
            tail = " | ".join(self._stderr_tail[-8:])
            raise TimeoutError(f"glyph worker not ready in {timeout:.0f}s: {tail}")

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    def rewrite(self, text: str, timeout: float = 60.0) -> dict:
        if not text or not text.strip():
            return {"ok": True, "text": text or "", "latency_ms": 0.0}
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(self.load_error or "glyph worker is not running")
        req_id = uuid.uuid4().hex[:12]
        box: list = []
        with self._cv:
            self._pending[req_id] = box
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"id": req_id, "text": text}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + timeout
            while not box and time.time() < deadline:
                self._cv.wait(timeout=0.2)
            self._pending.pop(req_id, None)
        if not box:
            raise TimeoutError("glyph rewrite timed out")
        msg = box[0]
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("error") or "glyph rewrite failed")
        return msg

    def _read_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("ready"):
                self._ready.set()
                continue
            req_id = msg.get("id")
            with self._cv:
                box = self._pending.get(req_id)
                if box is not None:
                    box.append(msg)
                self._cv.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            text = line.rstrip()
            if not text:
                continue
            self._stderr_tail.append(text)
            self._stderr_tail[:] = self._stderr_tail[-40:]
            print(f"[glyph] {text}", flush=True)


def _load(base: Path, adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def _generate(tok, model, text: str, max_new_tokens: int) -> str:
    import torch

    user = INSTRUCTION.format(asr=text)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
        )
    pred = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return pred.strip()


def serve(base: Path, adapter: Path, max_new_tokens: int) -> None:
    tok, model = _load(base, adapter)
    # One warmup so the first live final is not paying kernel init.
    _generate(tok, model, "你好。", max_new_tokens=16)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req.get("text") or ""
            t0 = time.perf_counter()
            if not text.strip():
                pred, dt = text, 0.0
            else:
                pred = _generate(tok, model, text, max_new_tokens)
                dt = (time.perf_counter() - t0) * 1000.0
            print(
                json.dumps(
                    {
                        "id": req.get("id"),
                        "ok": True,
                        "text": pred,
                        "latency_ms": round(dt, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            req_id = None
            try:
                req_id = json.loads(line).get("id")
            except Exception:
                pass
            print(
                json.dumps(
                    {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )


def bench(base: Path, adapter: Path, max_new_tokens: int, repeats: int) -> None:
    tok, model = _load(base, adapter)
    _generate(tok, model, "你好。", max_new_tokens=16)
    print(f"{'ms':>8}  text", flush=True)
    all_ms: list[float] = []
    for text in BENCH_TEXTS:
        times: list[float] = []
        pred = ""
        for _ in range(repeats):
            t0 = time.perf_counter()
            pred = _generate(tok, model, text, max_new_tokens)
            times.append((time.perf_counter() - t0) * 1000.0)
        all_ms.extend(times)
        mid = sorted(times)[len(times) // 2]
        print(
            f"{mid:8.1f}  {text}  =>  {pred}  (min {min(times):.1f} max {max(times):.1f})",
            flush=True,
        )
    ordered = sorted(all_ms)
    print(
        f"n={len(all_ms)} median={ordered[len(ordered)//2]:.1f}ms "
        f"min={ordered[0]:.1f}ms max={ordered[-1]:.1f}ms",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    if args.bench:
        bench(args.base, args.adapter, args.max_new_tokens, args.repeats)
        return 0
    if args.serve:
        serve(args.base, args.adapter, args.max_new_tokens)
        return 0
    ap.error("pass --serve or --bench")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
