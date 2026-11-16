#!/usr/bin/env python3
"""Local page: MiniCPM5 S2 route demo (mix_llmcont + b_ms_llm_cont).

对照保留。现网 FastLM=Step-Audio / SlowLM=Qwen3.5-2B 编排见
`STEP_ROUTE.md` 与 `python -m ranAudio.step_route`。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import librosa
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import ranAudio._boot  # noqa: F401

from ranAudio.model import MiniCPM5Audio, load_llm
from ranAudio.paths import MINICPM5, MINICPMO, RUNS_DIR, S2_TEST_JSONL, SENSEVOICE, resolve_wav
from ranAudio.protocol import BASE, FC, NOFC, S2_SYSTEM
from ranAudio.s2_catalog import read_jsonl

S15_CKPT = RUNS_DIR / "s15_asr" / "b_ms_llm_cont"
S15_ENC = S15_CKPT / "encoder_best.pt"
S15_PROJ = S15_CKPT / "projector_best.pt"
S2_CKPT = RUNS_DIR / "s2_fc_nofc_full" / "mix_llmcont"
S2_LLM = S2_CKPT / "llm_final.safetensors"
SAMPLES = ("s2t_fc_000", "s2t_chat_001", "s2t_asr_000")
MAX_BYTES = 15 * 1024 * 1024

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>S2 路由体验</title>
<style>
  :root {
    --bg: #f4f1ea;
    --ink: #1c1915;
    --muted: #6b645b;
    --line: #ddd6cb;
    --card: #fffdf8;
    --fc: #9a3412;
    --fc-bg: #ffedd5;
    --nofc: #166534;
    --nofc-bg: #dcfce7;
    --other: #44403c;
    --other-bg: #e7e5e4;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Songti SC", serif;
    background: var(--bg);
    color: var(--ink);
  }
  main { max-width: 720px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 32px; font-weight: 600; margin: 0 0 8px; }
  .sub { color: var(--muted); margin: 0 0 28px; line-height: 1.5; }
  .panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px;
  }
  .drop {
    border: 1.5px dashed #c4b8a8;
    border-radius: 12px;
    padding: 28px 16px;
    text-align: center;
    color: var(--muted);
    background: #faf7f2;
  }
  .drop.over { border-color: var(--ink); color: var(--ink); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
  button, .file {
    font: 15px/1.2 "Segoe UI", "PingFang SC", sans-serif;
    border-radius: 999px;
    border: 1px solid var(--ink);
    background: var(--ink);
    color: #fff;
    padding: 10px 16px;
    cursor: pointer;
  }
  button.ghost, .file { background: transparent; color: var(--ink); }
  button:disabled { opacity: 0.45; cursor: wait; }
  input[type=file] { display: none; }
  .samples { margin-top: 18px; }
  .samples p { margin: 0 0 8px; color: var(--muted); font-size: 14px; }
  audio { width: 100%; margin-top: 16px; }
  #result { margin-top: 18px; }
  .badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font: 13px/1.4 "Segoe UI", "PingFang SC", sans-serif;
    margin-bottom: 10px;
  }
  .fc { background: var(--fc-bg); color: var(--fc); }
  .nofc { background: var(--nofc-bg); color: var(--nofc); }
  .other, .asr { background: var(--nofc-bg); color: var(--nofc); }
  .body { font-size: 22px; line-height: 1.55; margin: 0; white-space: pre-wrap; }
  .raw {
    margin-top: 14px;
    color: var(--muted);
    font: 13px/1.5 ui-monospace, "SFMono-Regular", Menlo, monospace;
    word-break: break-all;
  }
  .err { color: #991b1b; margin-top: 12px; }
  .hint { color: var(--muted); font-size: 14px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <h1>S2 路由体验</h1>
  <p class="sub">上传或录一段话，看 <code>&lt;|FC|&gt;</code> / <code>&lt;|NOFC|&gt;</code>。权重：LLM <code>mix_llmcont</code> + 编码侧 <code>b_ms_llm_cont</code> best（相对原 mix_boundary：路由保持 1.0，听写 CER 0.267→0.259）。</p>
  <section class="panel">
    <div class="drop" id="drop">把音频拖到这里，或从下面选</div>
    <div class="row">
      <label class="file">选择文件<input id="file" type="file" accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg" /></label>
      <button class="ghost" id="rec" type="button">开始录音</button>
      <button class="ghost" id="stop" type="button" disabled>停止</button>
    </div>
    <div class="samples">
      <p>也可以先听这三条样例</p>
      <div class="row" id="samples"></div>
    </div>
    <audio id="player" controls hidden></audio>
    <p class="hint" id="status"></p>
    <p class="err" id="err" hidden></p>
    <div id="result" hidden>
      <div class="badge" id="badge"></div>
      <p class="body" id="body"></p>
      <div class="raw" id="raw"></div>
    </div>
  </section>
</main>
<script>
const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const recBtn = document.getElementById("rec");
const stopBtn = document.getElementById("stop");
const player = document.getElementById("player");
const statusEl = document.getElementById("status");
const errEl = document.getElementById("err");
const resultEl = document.getElementById("result");
const badge = document.getElementById("badge");
const bodyEl = document.getElementById("body");
const rawEl = document.getElementById("raw");
const samplesEl = document.getElementById("samples");
let recorder = null;
let chunks = [];
let busy = false;

function setBusy(on, text) {
  busy = on;
  recBtn.disabled = on || (recorder && recorder.state === "recording");
  fileInput.disabled = on;
  statusEl.textContent = text || "";
  for (const b of samplesEl.querySelectorAll("button")) b.disabled = on;
}

function showError(msg) {
  errEl.hidden = !msg;
  errEl.textContent = msg || "";
}

function showAudio(blob) {
  player.hidden = false;
  player.src = URL.createObjectURL(blob);
}

function showResult(data) {
  resultEl.hidden = false;
  badge.className = "badge " + (data.route || "other");
  badge.textContent = data.label || "未识别";
  bodyEl.textContent = data.body || "（空）";
  rawEl.textContent = data.raw || "";
}

async function inferBlob(blob, name) {
  showError("");
  resultEl.hidden = true;
  setBusy(true, "正在识别…");
  showAudio(blob);
  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream", "X-Filename": name || "audio.bin" },
      body: blob,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "识别失败");
    showResult(data);
    statusEl.textContent = "完成";
  } catch (err) {
    showError(err.message || String(err));
    statusEl.textContent = "";
  } finally {
    setBusy(false);
  }
}

async function inferSample(id, label) {
  showError("");
  resultEl.hidden = true;
  player.hidden = true;
  setBusy(true, "正在识别「" + label + "」…");
  try {
    const res = await fetch("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "识别失败");
    showResult(data);
    statusEl.textContent = "样例：" + (data.user || label);
  } catch (err) {
    showError(err.message || String(err));
    statusEl.textContent = "";
  } finally {
    setBusy(false);
  }
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  if (f) inferBlob(f, f.name);
});
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) inferBlob(f, f.name);
});

recBtn.addEventListener("click", async () => {
  showError("");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    stopBtn.disabled = true;
    recBtn.disabled = false;
    inferBlob(blob, "recording.webm");
  };
  recorder.start();
  recBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "正在录音…";
});
stopBtn.addEventListener("click", () => { if (recorder && recorder.state === "recording") recorder.stop(); });

fetch("/api/samples").then((r) => r.json()).then((rows) => {
  for (const row of rows) {
    const b = document.createElement("button");
    b.className = "ghost";
    b.type = "button";
    b.textContent = row.user;
    b.addEventListener("click", () => inferSample(row.id, row.user));
    samplesEl.appendChild(b);
  }
});
</script>
</body>
</html>
"""


def present(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith(f"{FC}{BASE}"):
        body = raw[len(FC) + len(BASE) :].strip()
        return {"route": "fc", "label": "FC · 上抛", "body": body, "raw": raw}
    if raw.startswith(NOFC):
        body = raw[len(NOFC) :].strip()
        return {"route": "nofc", "label": "NOFC · 本地", "body": body, "raw": raw}
    return {"route": "other", "label": "其它", "body": raw, "raw": raw}


def load_audio(path: Path) -> np.ndarray:
    try:
        wav, _ = librosa.load(str(path), sr=16000, mono=True)
        if wav.size:
            return wav
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        dst = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(dst)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav, _ = librosa.load(str(dst), sr=16000, mono=True)
    finally:
        dst.unlink(missing_ok=True)
    if wav.size == 0:
        raise RuntimeError("音频是空的")
    return wav


class Demo:
    def __init__(self, model: MiniCPM5Audio, samples: list[dict]):
        self.model = model
        self.samples = samples
        self.lock = threading.Lock()

    def run_wav(self, wav: np.ndarray) -> dict:
        with self.lock:
            text = self.model.generate_text(wav, tools=None, system=S2_SYSTEM, max_new_tokens=128)
        return present(text)

    def run_path(self, path: Path) -> dict:
        return self.run_wav(load_audio(path))


def load_model(
    encoder: Path,
    projector: Path,
    sensevoice: str,
    llm_weights: Path | None = None,
    tokenizer_dir: Path | None = None,
) -> MiniCPM5Audio:
    tokenizer = None
    if tokenizer_dir is not None and tokenizer_dir.is_dir():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        print(f"demo reuse tokenizer {tokenizer_dir}", flush=True)
    model = MiniCPM5Audio.from_pretrained(
        str(MINICPM5),
        str(MINICPMO),
        projector_path=projector,
        encoder_path=encoder,
        encoder_backend="sensevoice",
        sensevoice_path=sensevoice or SENSEVOICE,
        tokenizer=tokenizer,
    )
    if llm_weights is not None and llm_weights.is_file():
        print(f"loading llm weights {llm_weights}", flush=True)
        load_llm(model, llm_weights)
    model.eval()
    return model


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"[demo] {self.address_string()} {fmt % args}", flush=True)

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/samples":
                self._json(
                    200,
                    [{"id": r["id"], "user": r.get("user") or r["id"]} for r in demo.samples],
                )
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/api/infer":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n <= 0 or n > MAX_BYTES:
                        self._json(400, {"error": "音频为空或超过 15MB"})
                        return
                    data = self.rfile.read(n)
                    name = self.headers.get("X-Filename") or "audio.bin"
                    suffix = Path(name).suffix.lower() or ".bin"
                    if suffix not in {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".bin"}:
                        suffix = ".bin"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(data)
                        path = Path(tmp.name)
                    try:
                        out = demo.run_path(path)
                    finally:
                        path.unlink(missing_ok=True)
                    self._json(200, out)
                    return
                if self.path == "/api/sample":
                    n = int(self.headers.get("Content-Length") or 0)
                    req = json.loads(self.rfile.read(n) or b"{}")
                    sid = str(req.get("id") or "")
                    row = next((r for r in demo.samples if r["id"] == sid), None)
                    if row is None:
                        self._json(404, {"error": "没有这条样例"})
                        return
                    out = demo.run_path(resolve_wav(row["wav"]))
                    out["user"] = row.get("user")
                    out["id"] = sid
                    self._json(200, out)
                    return
                self._json(404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--encoder", default=str(S15_ENC))
    ap.add_argument("--projector", default=str(S15_PROJ))
    ap.add_argument("--llm-weights", default=str(S2_LLM))
    ap.add_argument("--tokenizer-dir", default=str(S2_CKPT / "tokenizer"))
    ap.add_argument("--sensevoice", default="")
    args = ap.parse_args()

    rows = {r["id"]: r for r in read_jsonl(S2_TEST_JSONL)}
    samples = [rows[sid] for sid in SAMPLES if sid in rows]
    llm_w = Path(args.llm_weights) if args.llm_weights else None
    tok = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    print(
        f"loading s2 demo encoder={args.encoder} projector={args.projector} llm={llm_w}",
        flush=True,
    )
    model = load_model(
        Path(args.encoder),
        Path(args.projector),
        args.sensevoice,
        llm_weights=llm_w,
        tokenizer_dir=tok,
    )
    demo = Demo(model, samples)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    print(f"demo ready http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
