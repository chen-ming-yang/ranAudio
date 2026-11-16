#!/usr/bin/env python3
"""L1 streaming ASR demo server: WebSocket PCM in → PartialHypothesis out + StableRenderer UI.

  PYTHON=/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python
  $PYTHON scripts/l1_server.py --port 7863
  $PYTHON scripts/l1_server.py --fake --port 7863   # UI only, no GPU

WS protocol:
  client → {"type":"start","sample_rate":16000,"language":"Chinese"}
  client → binary int16 LE PCM frames (mono 16k)
  client → {"type":"end"}
  server → PartialHypothesis JSON (type partial|final)
  server → {"type":"postprocess","text","via":"llm_postprocess","latency_ms"}
  server → {"type":"error","message":"..."}

Only final text is rewritten. Partials are not.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from glyph_postprocess import (  # noqa: E402
    DEFAULT_ADAPTER,
    DEFAULT_BASE,
    DEFAULT_PYTHON,
    GlyphRewriteClient,
)
from qwen_streaming_engine import (  # noqa: E402
    DEFAULT_MODEL,
    FakeStreamingEngine,
    QwenStreamingEngine,
    SAMPLE_RATE,
)

ROOT = Path(__file__).resolve().parents[1]


def _pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros((0,), dtype=np.float32)
    n = len(data) // 2
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    ints = np.frombuffer(data, dtype="<i2", count=n)
    return (ints.astype(np.float32) / 32768.0)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kimi L1 · 流式 ASR</title>
<style>
  :root {
    --bg: #f6f3ee;
    --ink: #1a1a1a;
    --muted: #8a8a8a;
    --accent: #0b6e4f;
    --card: #fffdf9;
    --line: #e4ddd3;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh;
    font-family: "Source Han Sans SC", "Noto Sans SC", "PingFang SC", sans-serif;
    background:
      radial-gradient(900px 420px at 10% -10%, #e8f2ec 0%, transparent 55%),
      radial-gradient(700px 380px at 100% 0%, #f0e6d8 0%, transparent 50%),
      var(--bg);
    color: var(--ink);
  }
  main { max-width: 720px; margin: 0 auto; padding: 28px 18px 64px; }
  h1 { font-size: 1.45rem; font-weight: 650; letter-spacing: 0.02em; margin: 0 0 6px; }
  .sub { color: var(--muted); font-size: 0.92rem; margin-bottom: 22px; }
  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 16px 16px 14px; box-shadow: 0 10px 30px rgba(0,0,0,.04);
  }
  #draft {
    min-height: 140px; padding: 14px 14px 18px;
    border: 1px solid var(--line); border-radius: 10px;
    background: #fff; line-height: 1.65; font-size: 1.15rem;
    outline: none; white-space: pre-wrap; word-break: break-word;
  }
  #draft:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(11,110,79,.12); }
  .stable { color: var(--ink); font-style: normal; }
  .provisional { color: var(--muted); font-style: italic; }
  .caret {
    display: inline-block; width: 2px; height: 1.1em; margin-left: 1px;
    background: var(--accent); vertical-align: text-bottom;
    animation: blink 1s step-end infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
  .row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; align-items: center; }
  button {
    appearance: none; border: 0; border-radius: 999px; padding: 10px 18px;
    font-size: 0.95rem; font-weight: 600; cursor: pointer;
    background: var(--accent); color: #fff;
  }
  button.secondary { background: #ece7df; color: var(--ink); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  #recBtn.recording { background: #b42318; }
  .meta { font-size: 0.82rem; color: var(--muted); margin-top: 10px; min-height: 1.2em; }
  .log {
    margin-top: 18px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.75rem; color: #555; max-height: 180px; overflow: auto;
    background: #faf7f2; border-radius: 8px; padding: 10px; border: 1px solid var(--line);
  }
  .badge {
    display: inline-block; font-size: 0.72rem; padding: 2px 8px; border-radius: 999px;
    background: #e8f2ec; color: var(--accent); font-weight: 600;
  }
</style>
</head>
<body>
<main>
  <h1>流式听写 <span class="badge">L1</span></h1>
  <p class="sub">Qwen3-ASR-1.7B partial → StableRenderer · 灰斜体为 provisional</p>
  <div class="panel">
    <div id="draft" contenteditable="true" role="textbox" aria-label="听写草稿" spellcheck="false"></div>
    <div class="row">
      <button type="button" id="recBtn">按住说话 / 点击开始</button>
      <button type="button" class="secondary" id="clearBtn">清空</button>
      <span id="readyBadge" class="badge">connecting…</span>
    </div>
    <div class="meta" id="meta"></div>
  </div>
  <div class="log" id="log"></div>
</main>
<script>
class StableRenderer {
  constructor(rootEl) {
    this.root = rootEl;
    this.stable = "";
    this.provisional = "";
    this.userEditing = false;
    this._listeningFocus = false;
    rootEl.addEventListener("focus", () => { this._listeningFocus = true; });
    rootEl.addEventListener("blur", () => {
      this._listeningFocus = false;
      this.userEditing = false;
      this._syncUserStableFromDom();
    });
    rootEl.addEventListener("input", () => {
      if (this._listeningFocus) this.userEditing = true;
    });
  }

  _syncUserStableFromDom() {
    // On blur, treat full visible text as stable baseline
    const t = this.root.innerText.replace(/\u200b/g, "").replace(/\n$/, "");
    this.stable = t;
    this.provisional = "";
  }

  _setHtml(stable, provisional, showCaret) {
    const esc = (s) => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    let html = `<span class="stable">${esc(stable)}</span>`;
    if (provisional)
      html += `<span class="provisional">${esc(provisional)}</span>`;
    if (showCaret) html += `<span class="caret"></span>`;
    this.root.innerHTML = html || `<span class="provisional"></span><span class="caret"></span>`;
  }

  onPartial({ text, stable_len, is_final }) {
    const t = text || "";
    const sl = Math.max(0, Math.min(t.length, stable_len|0));
    if (this.userEditing) {
      // Keep user stable; only refresh provisional tail from model
      const prov = t.slice(sl);
      this.provisional = prov;
      this._setHtml(this.stable, this.provisional, true);
      return;
    }
    const nextStable = t.slice(0, sl);
    const nextProv = t.slice(sl);
    this.stable = nextStable;
    this.provisional = nextProv;
    if (is_final) {
      this.stable = t;
      this.provisional = "";
      this._setHtml(this.stable, "", false);
      return;
    }
    this._setHtml(this.stable, this.provisional, true);
  }

  clear() {
    this.stable = "";
    this.provisional = "";
    this.userEditing = false;
    this.root.innerHTML = "";
  }

  fullText() {
    return (this.stable || "") + (this.provisional || "");
  }

  applyFinalText(text) {
    if (this.userEditing) return false;
    this.stable = text || "";
    this.provisional = "";
    this._setHtml(this.stable, "", false);
    return true;
  }
}

const draftEl = document.getElementById("draft");
const recBtn = document.getElementById("recBtn");
const clearBtn = document.getElementById("clearBtn");
const metaEl = document.getElementById("meta");
const logEl = document.getElementById("log");
const readyBadge = document.getElementById("readyBadge");
const renderer = new StableRenderer(draftEl);

function logLine(...args) {
  const line = args.map(String).join(" ");
  const div = document.createElement("div");
  div.textContent = new Date().toLocaleTimeString() + "  " + line;
  logEl.prepend(div);
}

async function refreshReady() {
  try {
    const r = await fetch("/api/ready");
    const d = await r.json();
    readyBadge.textContent = d.ready
      ? ("ready · " + (d.engine || "?"))
      : ("loading · " + (d.load_error || ""));
  } catch (e) {
    readyBadge.textContent = "offline";
  }
}
refreshReady();
setInterval(refreshReady, 4000);

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return proto + "//" + location.host + "/ws/asr";
}

let audioCtx = null;
let mediaStream = null;
let processor = null;
let source = null;
let ws = null;
let recording = false;

function setRecording(on) {
  recording = on;
  recBtn.classList.toggle("recording", on);
  recBtn.textContent = on ? "点击结束" : "按住说话 / 点击开始";
}

function downsampleTo16k(float32, inRate) {
  if (inRate === 16000) return float32;
  const ratio = inRate / 16000;
  const outLen = Math.floor(float32.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = Math.floor(i * ratio);
    out[i] = float32[idx];
  }
  return out;
}

function floatTo16BitPCM(float32) {
  const buf = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

async function startRec() {
  if (recording) return;
  const ready = await fetch("/api/ready").then(r => r.json());
  if (!ready.ready) {
    metaEl.textContent = "引擎未就绪: " + (ready.load_error || "");
    return;
  }
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  source = audioCtx.createMediaStreamSource(mediaStream);
  // ScriptProcessor is deprecated but widely available; 4096 ≈ 85ms @48k
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";

  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error("WebSocket 连接失败"));
  });

  ws.send(JSON.stringify({ type: "start", sample_rate: 16000, language: "Chinese" }));
  logLine("ws start");

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "error") {
        metaEl.textContent = "错误: " + msg.message;
        logLine("error", msg.message);
        return;
      }
      if (msg.type === "partial" || msg.type === "final") {
        renderer.onPartial(msg);
        metaEl.textContent =
          (msg.is_final ? "final" : "partial") +
          " · stable=" + msg.stable_len +
          " · " + (msg.audio_ms || 0) + "ms" +
          " · " + (msg.engine || "");
        if (msg.is_final) logLine("final", JSON.stringify(msg.text));
        return;
      }
      if (msg.type === "postprocess") {
        const ms = msg.latency_ms != null ? msg.latency_ms : "?";
        if (msg.ok === false) {
          metaEl.textContent = "后处理失败，保留识别结果 · " + (msg.error || "");
          logLine("postprocess fail", msg.error || "");
          return;
        }
        renderer.applyFinalText(msg.text || "");
        metaEl.textContent = "postprocess · " + ms + "ms";
        logLine("postprocess", ms + "ms", JSON.stringify(msg.text));
      }
    } catch (e) {
      logLine("bad msg", e.message || e);
    }
  };

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== 1) return;
    const input = e.inputBuffer.getChannelData(0);
    const down = downsampleTo16k(input, audioCtx.sampleRate);
    ws.send(floatTo16BitPCM(down));
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
  setRecording(true);
  metaEl.textContent = "录音中…";
}

async function stopRec() {
  if (!recording) return;
  setRecording(false);
  try {
    if (processor) { processor.disconnect(); processor.onaudioprocess = null; }
    if (source) source.disconnect();
    if (audioCtx) await audioCtx.close();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  } catch (_) {}
  processor = source = audioCtx = mediaStream = null;
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: "end" }));
    logLine("ws end");
  }
}

recBtn.addEventListener("click", () => {
  if (recording) stopRec();
  else startRec().catch(err => {
    metaEl.textContent = err.message || String(err);
    logLine("rec fail", err.message || err);
    setRecording(false);
  });
});
clearBtn.addEventListener("click", () => {
  renderer.clear();
  metaEl.textContent = "";
});
</script>
</body>
</html>
"""


def build_engine(args: argparse.Namespace):
    if args.fake:
        eng = FakeStreamingEngine()
        eng.load()
        return eng
    eng = QwenStreamingEngine(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        language=args.language,
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_size_sec,
    )
    return eng


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("KIMI_L1_PORT", "7863")))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fake", action="store_true", help="UI-only fake engine")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--unfixed-chunk-num", type=int, default=2)
    ap.add_argument("--unfixed-token-num", type=int, default=5)
    ap.add_argument("--chunk-size-sec", type=float, default=1.0)
    ap.add_argument(
        "--llm-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rewrite each ASR final with the glyph LoRA (default on)",
    )
    ap.add_argument("--llm-python", type=Path, default=DEFAULT_PYTHON)
    ap.add_argument("--llm-base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--llm-adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--llm-max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    engine = build_engine(args)
    load_error_box: dict[str, str] = {"err": ""}
    rewriter: GlyphRewriteClient | None = None

    # Load on main thread BEFORE uvicorn so vLLM does not hit
    # "CUDA already initialized → spawn" pitfalls.
    try:
        if not args.fake:
            print("[l1] loading Qwen3-ASR (vLLM, enforce_eager)…", flush=True)
        engine.load()
        print("[l1] engine ready", flush=True)
    except Exception as exc:  # noqa: BLE001
        load_error_box["err"] = f"{type(exc).__name__}: {exc}"
        print("[l1] load failed:", load_error_box["err"], flush=True)
        engine.ready = False
        engine.load_error = load_error_box["err"]

    if args.llm_postprocess and engine.ready:
        rewriter = GlyphRewriteClient(
            python=args.llm_python,
            base=args.llm_base,
            adapter=args.llm_adapter,
            max_new_tokens=args.llm_max_new_tokens,
        )
        try:
            print("[l1] loading glyph LoRA worker…", flush=True)
            rewriter.start()
            print("[l1] glyph worker ready", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[l1] glyph worker failed, finals stay raw: {exc}", flush=True)
            rewriter.close()
            rewriter = None

    app = FastAPI(title="Kimi L1 Streaming ASR")
    # threading lock: Qwen streaming is single-stream; avoid asyncio.Lock() before loop starts
    engine_lock = threading.Lock()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.get("/api/ready")
    async def ready() -> JSONResponse:
        return JSONResponse(
            {
                "ready": bool(getattr(engine, "ready", False)),
                "engine": getattr(engine, "ENGINE_NAME", "?"),
                "load_error": getattr(engine, "load_error", "") or load_error_box.get("err", ""),
                "model": args.model if not args.fake else "fake",
                "chunk_size_sec": getattr(engine, "chunk_size_sec", args.chunk_size_sec),
                "llm_postprocess": rewriter is not None,
                "routes": [getattr(r, "path", None) for r in app.routes],
            }
        )

    async def _emit_postprocess(ws: WebSocket, session_id: str, text: str) -> None:
        if rewriter is None:
            return
        try:
            result = await asyncio.to_thread(rewriter.rewrite, text)
            await ws.send_json(
                {
                    "type": "postprocess",
                    "session_id": session_id,
                    "text": result.get("text", text),
                    "via": "llm_postprocess",
                    "latency_ms": result.get("latency_ms", 0),
                    "ok": True,
                }
            )
            print(
                f"[l1] postprocess {result.get('latency_ms', 0)}ms session={session_id}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[l1] postprocess failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                await ws.send_json(
                    {
                        "type": "postprocess",
                        "session_id": session_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "via": "llm_postprocess",
                    }
                )
            except Exception:
                pass

    def _start(lang: str) -> str:
        with engine_lock:
            return engine.start_session(language=lang)

    def _feed(pcm: np.ndarray):
        with engine_lock:
            return engine.feed_pcm16k(pcm)

    def _finish():
        with engine_lock:
            return engine.finish_session()

    # Use Starlette websocket_route (no FastAPI DI) — @app.websocket DI can
    # raise WebSocketRequestValidationError before accept → HTTP 403.
    @app.websocket_route("/ws/asr")
    async def ws_asr(ws: WebSocket) -> None:
        print("[l1] ws handler enter", flush=True)
        await ws.accept()
        print("[l1] ws accepted", flush=True)
        session_started = False
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "text" in message and message["text"] is not None:
                    try:
                        msg = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "invalid json"})
                        continue
                    mtype = msg.get("type")
                    if mtype == "start":
                        if not engine.ready:
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "message": engine.load_error or "engine not ready",
                                }
                            )
                            continue
                        lang = msg.get("language") or args.language
                        sid = await asyncio.to_thread(_start, lang)
                        session_started = True
                        await ws.send_json(
                            {
                                "type": "session",
                                "session_id": sid,
                                "engine": engine.ENGINE_NAME,
                            }
                        )
                    elif mtype == "end":
                        if session_started:
                            hyp = await asyncio.to_thread(_finish)
                            await ws.send_json(hyp.to_dict())
                            session_started = False
                            if hyp.is_final:
                                await _emit_postprocess(ws, hyp.session_id, hyp.text)
                        break
                    else:
                        await ws.send_json(
                            {"type": "error", "message": f"unknown type {mtype}"}
                        )
                elif "bytes" in message and message["bytes"] is not None:
                    if not session_started:
                        await ws.send_json(
                            {"type": "error", "message": "send start before pcm"}
                        )
                        continue
                    pcm = _pcm16_bytes_to_float32(message["bytes"])
                    hyp = await asyncio.to_thread(_feed, pcm)
                    if hyp is not None:
                        await ws.send_json(hyp.to_dict())
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[l1] ws error: {type(exc).__name__}: {exc}", flush=True)
            try:
                await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            if session_started:
                try:
                    await asyncio.to_thread(_finish)
                except Exception:
                    pass

    print(f"[l1] serving http://{args.host}:{args.port}/  fake={args.fake}", flush=True)
    print(f"[l1] routes={[getattr(r,'path',None) for r in app.routes]}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
