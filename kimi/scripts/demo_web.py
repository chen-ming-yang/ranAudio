#!/usr/bin/env python3
"""Local voice-edit demo: SenseVoice ASR -> (rules / Kimi LLM) rewrite, plus optional Q&A.

Stage 1 (转写 / 指令识别): SenseVoice-Small via FunASR.
Stage 2 (改写): rule engine first; Kimi-Audio LLM is lazy-loaded only when needed.
Optional --speech loads detokenizer for spoken replies (heavy VRAM).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import wave
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from sensevoice_asr import DEFAULT_SENSEVOICE, SenseVoiceASR  # noqa: E402
from voice_edit_lib import (  # noqa: E402
    apply_edit_rules as _apply_edit_rules,
    apply_fuzzy_replace as _apply_fuzzy_replace,
    apply_radical_describe_replace as _apply_radical_describe_replace,
    apply_simple_replace as _apply_simple_replace,
    apply_toggle as _apply_toggle,
    clean_rewrite as _clean_rewrite,
    clean_spoken2written as _clean_spoken2written,
    has_name_clarify as _has_name_clarify,
    is_spoken2written_instruction as _is_spoken2written_instruction,
    looks_like_bad_spoken2written as _looks_like_bad_spoken2written,
    looks_like_garbage as _looks_like_garbage,
    normalize_display as _normalize_display,
    normalize_list_structure as _normalize_list_structure,
    resolve_name_clarify as _resolve_name_clarify,
    strip_fillers as _strip_fillers,
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")


def _parse_truthy(val: object | None) -> bool | None:
    """Parse request flag; None if absent / unknown."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "y"):
        return True
    if s in ("0", "false", "no", "off", "n", ""):
        return False
    return None

ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = ROOT / "models" / "Kimi-Audio-7B-Instruct"
TEST_AUDIOS = ROOT / "test_audios"
ERROR_SAMPLES = ROOT / "error_samples"
ERROR_CASES = ERROR_SAMPLES / "cases.jsonl"
ERROR_WAVS = ERROR_SAMPLES / "wavs"
MAX_BYTES = 20 * 1024 * 1024
MAX_DRAFT_CHARS = 8000
MAX_NOTE_CHARS = 2000
LOG_BUF: deque[str] = deque(maxlen=400)
LOG_LOCK = threading.Lock()
LOG_FILE = ROOT / "demo_app.log"
ERROR_LOCK = threading.Lock()


def _slog(msg: str, *args) -> None:
    line = msg % args if args else msg
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    full = f"[{stamp}] {line}"
    with LOG_LOCK:
        LOG_BUF.append(full)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(full + "\n")
        except OSError:
            pass
    print(full, flush=True)
    sys.stderr.flush()

SAMPLES = [
    {"id": "asr", "label": "转写样例", "path": "asr_example.wav", "mode": "asr"},
    {"id": "qa", "label": "问答样例", "path": "qa_example.wav", "mode": "chat"},
]

SAMPLING = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

EDIT_SAMPLING = {
    **SAMPLING,
    "text_repetition_penalty": 1.15,
    "text_repetition_window_size": 8,
    "max_new_tokens": 64,
}

# Spoken→written: short decode + stronger anti-loop (model tends to emit 2–3 candidates)
S2W_SAMPLING = {
    **EDIT_SAMPLING,
    "text_repetition_penalty": 1.35,
    "text_repetition_window_size": 6,
    "max_new_tokens": 48,
}

ASR_PROMPT = "请将音频内容转换为文字。"

# 改写必须走「先 ASR 指令，再纯文本改写」：带音频时模型易退化成转写指令本身。
EDIT_REWRITE_PROMPT = (
    "你是文本改写器。根据【修改指令】改写【草稿】，输出改写后的完整句子。\n"
    "规则：只输出改写结果；不要解释；不要复述或改写指令本身；"
    "未提到的字原样保留。\n"
    "示例1：草稿「打开窗帘」+ 指令「把窗帘变成服务器」→ 打开服务器\n"
    "示例2：草稿「今天去上海」+ 指令「把上海改成北京」→ 今天去北京\n"
    "示例3：草稿「请把灯关掉」+ 指令「改成打开」→ 请把灯打开\n"
    "【草稿】{draft}\n"
    "【修改指令】{instruction}\n"
    "【改写结果】"
)

# 转写后：句内「X是…的X」姓名字形说明 → 纠正姓名与说明中的错字
NAME_CLARIFY_PROMPT = (
    "用户口述了名字，并用「X是…的X」说明正确字形（组词或偏旁）。\n"
    "请输出纠正后的完整句子：按说明把姓名和说明里的错字改成对应正确字；"
    "保留原句结构与标点；只输出纠正后全文，不要解释。\n"
    "示例：原文「我叫陈明阳，名是名字的名，杨是提手旁的杨。」"
    "→ 我叫陈名扬，名是名字的名，扬是提手旁的扬。\n"
    "【原文】{text}\n"
    "【纠正】"
)

# 口语 → 书面（CS2W 风格）；故意强调/节奏重复须保留
SPOKEN2WRITTEN_PROMPT = (
    "你是中文口语转书面语改写器。将【原文】改成通顺的书面语。\n"
    "可做：去掉填充词（嗯/呃/那个）；去掉口误性半截重说（如「不要不要」→「不要」）；"
    "补缺词/删赘词；口语词改书面；明显 ASR 错字可改。\n"
    "不可做：改变原意；扩写；解释；删掉劝诫词「千万/一定/务必」；"
    "输出多个候选；把同一句改写两遍或三遍。\n"
    "重复：口误叠词可去一层；若像强调、口号、节奏、有意复读（如「好好好好」）则保留；"
    "拿不准时保留原重复。\n"
    "示例：原文「千万不要不要有贪婪的意识」→ 千万不要有贪婪的意识\n"
    "若原文已是简短指令或书面语，原样输出。\n"
    "只输出一句最终结果，不要解释，不要列举多种改法。\n"
    "【原文】{text}\n"
    "【书面语】"
)

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Kimi-Audio</title>
<link rel="preconnect" href="https://fonts.loli.net" crossorigin />
<link href="https://fonts.loli.net/css2?family=Syne:wght@600;700;800&family=Noto+Sans+SC:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  :root {
    --bg0: #0b1220;
    --bg1: #122033;
    --ink: #e8eef7;
    --muted: #8fa3bf;
    --line: rgba(232,238,247,0.14);
    --accent: #3dd6c6;
    --accent-dim: rgba(61,214,198,0.16);
    --warn: #f0a37a;
    --err: #ff7b7b;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    color: var(--ink);
    font-family: "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background:
      radial-gradient(900px 480px at 12% -10%, rgba(61,214,198,0.18), transparent 60%),
      radial-gradient(700px 420px at 95% 8%, rgba(80,120,220,0.16), transparent 55%),
      linear-gradient(165deg, var(--bg0), var(--bg1) 55%, #0a101a);
    min-height: 100vh;
  }
  main {
    max-width: 760px;
    margin: 0 auto;
    padding: clamp(36px, 8vh, 72px) 22px 80px;
  }
  .brand {
    font-family: Syne, "Noto Sans SC", sans-serif;
    font-weight: 800;
    font-size: clamp(42px, 8vw, 64px);
    letter-spacing: -0.03em;
    line-height: 0.95;
    margin: 0 0 14px;
  }
  .brand span { color: var(--accent); }
  .headline {
    font-family: Syne, "Noto Sans SC", sans-serif;
    font-weight: 600;
    font-size: clamp(20px, 3.4vw, 26px);
    margin: 0 0 10px;
    max-width: 18em;
  }
  .sub {
    margin: 0 0 28px;
    color: var(--muted);
    line-height: 1.55;
    max-width: 36em;
  }
  .modes {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }
  .modes button {
    border: 1px solid var(--line);
    background: transparent;
    color: var(--muted);
    border-radius: 999px;
    padding: 8px 14px;
    font: 500 14px/1 "Noto Sans SC", sans-serif;
    cursor: pointer;
  }
  .modes button.on {
    color: var(--bg0);
    background: var(--accent);
    border-color: var(--accent);
  }
  .stage {
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    backdrop-filter: blur(8px);
  }
  .stage + .stage { margin-top: 14px; }
  .stage h2 {
    font-family: Syne, "Noto Sans SC", sans-serif;
    font-size: 16px;
    font-weight: 600;
    margin: 0 0 12px;
  }
  .drop {
    border: 1.5px dashed rgba(232,238,247,0.28);
    border-radius: 14px;
    padding: 34px 16px;
    text-align: center;
    color: var(--muted);
    background: rgba(0,0,0,0.18);
  }
  .drop.compact { padding: 22px 14px; }
  .drop.over { border-color: var(--accent); color: var(--ink); background: var(--accent-dim); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
  button, .file {
    font: 500 14px/1.2 "Noto Sans SC", sans-serif;
    border-radius: 999px;
    border: 1px solid var(--ink);
    background: var(--ink);
    color: var(--bg0);
    padding: 10px 16px;
    cursor: pointer;
  }
  button.ghost, .file {
    background: transparent;
    color: var(--ink);
    border-color: var(--line);
  }
  button:disabled, .file:has(input:disabled) { opacity: 0.45; cursor: wait; }
  input[type=file] { display: none; }
  .samples { margin-top: 16px; }
  .samples p { margin: 0 0 8px; color: var(--muted); font-size: 13px; }
  audio { width: 100%; margin-top: 16px; }
  #status { color: var(--muted); font-size: 13px; margin: 12px 0 0; min-height: 1.2em; }
  #err { color: var(--err); margin-top: 10px; }
  .draft-box {
    width: 100%;
    min-height: 140px;
    margin-top: 8px;
    padding: 16px 18px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(0,0,0,0.28);
    color: var(--ink);
    font: 500 clamp(18px, 2.8vw, 24px)/1.55 "Noto Sans SC", sans-serif;
    resize: vertical;
    outline: none;
  }
  .draft-box:focus { border-color: rgba(61,214,198,0.55); }
  .draft-box::placeholder { color: rgba(143,163,191,0.65); font-weight: 400; font-size: 16px; }
  .meta {
    margin-top: 10px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
    min-height: 1.2em;
    white-space: pre-wrap;
  }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 12px; }
  .save-panel {
    margin-top: 14px;
    padding: 14px;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(0,0,0,0.22);
  }
  .save-panel h3 {
    margin: 0 0 8px;
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
  }
  .save-panel .hint { margin: 0 0 10px; color: var(--muted); font-size: 12px; line-height: 1.45; }
  .save-panel label.field {
    display: block;
    margin: 8px 0 4px;
    color: var(--muted);
    font-size: 12px;
  }
  .save-panel textarea, .save-panel input[type=text] {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(0,0,0,0.28);
    color: var(--ink);
    font: 400 14px/1.45 "Noto Sans SC", sans-serif;
  }
  .save-panel textarea { min-height: 72px; resize: vertical; }
  .save-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  button.warn {
    background: transparent;
    color: var(--warn);
    border-color: rgba(240,163,122,0.45);
  }
  button.danger {
    background: transparent;
    color: var(--err);
    border-color: rgba(255,123,123,0.45);
  }
  .ready {
    margin-top: 18px;
    color: var(--muted);
    font-size: 12px;
  }
  .ready.ok { color: var(--accent); }
  .ready.bad { color: var(--warn); }
  .mic-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin-top: 18px;
  }
  .mic-btn {
    width: 88px;
    height: 88px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    background: var(--accent-dim);
    color: var(--accent);
    font: 600 14px/1 "Noto Sans SC", sans-serif;
    cursor: pointer;
    padding: 0;
    box-shadow: 0 0 0 0 rgba(61,214,198,0.35);
  }
  .mic-btn.live {
    background: #e85d5d;
    border-color: #e85d5d;
    color: #fff;
    animation: pulse 1.2s ease-out infinite;
  }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(232,93,93,0.45); }
    70% { box-shadow: 0 0 0 18px rgba(232,93,93,0); }
    100% { box-shadow: 0 0 0 0 rgba(232,93,93,0); }
  }
  .mic-hint { color: var(--muted); font-size: 13px; text-align: center; }
  .opt-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin: 10px 0 0;
    color: var(--muted);
    font-size: 13px;
  }
  .opt-row label { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; }
  .opt-row input { accent-color: var(--accent); width: 15px; height: 15px; }
  .cmp-panel {
    margin-top: 16px;
    padding: 12px 14px;
    border: 1px dashed var(--line);
    border-radius: 12px;
    background: rgba(0,0,0,0.18);
  }
  .cmp-panel[hidden] { display: none !important; }
  .cmp-panel h3 {
    margin: 0 0 10px;
    font: 600 13px/1.3 "Noto Sans SC", sans-serif;
    color: var(--muted);
    letter-spacing: 0.02em;
  }
  .cmp-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  @media (max-width: 640px) {
    .cmp-grid { grid-template-columns: 1fr; }
  }
  .cmp-col label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--accent);
    font-weight: 600;
  }
  .cmp-col textarea {
    width: 100%;
    min-height: 88px;
    resize: vertical;
    box-sizing: border-box;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: rgba(11,18,32,0.65);
    color: var(--ink);
    font: 14px/1.5 "Noto Sans SC", sans-serif;
  }
  .cmp-meta { margin: 8px 0 0; font-size: 12px; color: var(--muted); }
  .logbox {
    margin-top: 22px;
    border-top: 1px solid var(--line);
    padding-top: 14px;
  }
  .logbox summary {
    cursor: pointer;
    color: var(--muted);
    font-size: 13px;
    user-select: none;
  }
  .logbox summary:hover { color: var(--ink); }
  #logView {
    margin: 10px 0 0;
    max-height: 280px;
    overflow: auto;
    padding: 12px;
    background: rgba(0,0,0,0.28);
    border: 1px solid var(--line);
    border-radius: 10px;
    color: #b8c7db;
    font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .log-actions { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
  .log-actions button { font-size: 12px; padding: 6px 12px; }
</style>
</head>
<body>
<main>
  <h1 class="brand">Kimi<span>-Audio</span></h1>
  <p class="headline">一个框：先说写入文字，再说就改上面的字。</p>
  <p class="sub">点麦克风录音。本机用 <strong>http://localhost:7861</strong>；局域网用 IP 时请开 <strong>https://</strong>（自签名点「继续访问」）。</p>
  <div class="modes">
    <button type="button" class="on" data-mode="asr" id="modeAsr">语音编辑</button>
    <button type="button" data-mode="chat" id="modeChat">语音问答</button>
  </div>
  <section class="stage">
    <h2 id="stageTitle">当前文本</h2>
    <textarea id="draftBox" class="draft-box" placeholder="还没有文字。先点麦克风说一段，转写会出现在这里；再说修改指令就会改这段文字。" rows="5"></textarea>
    <p class="meta" id="meta"></p>
    <div class="cmp-panel" id="cmpPanel" hidden>
      <h3>调试对照（转写 vs 书面化）· 当前文本以书面化结果为准</h3>
      <div class="cmp-grid">
        <div class="cmp-col">
          <label for="asrTextDbg">① 转写结果</label>
          <textarea id="asrTextDbg" readonly placeholder="SenseVoice + 规则后、书面化前"></textarea>
        </div>
        <div class="cmp-col">
          <label for="writtenTextDbg">② 书面化结果</label>
          <textarea id="writtenTextDbg" readonly placeholder="未开书面化或未跑 LLM"></textarea>
        </div>
      </div>
      <p class="cmp-meta" id="cmpMeta"></p>
    </div>
    <div class="mic-wrap">
      <button type="button" class="mic-btn" id="rec" aria-label="麦克风录音">按住说</button>
      <p class="mic-hint" id="micHint">空框 → 转写；有字 → 按语音修改</p>
      <div class="opt-row">
        <label for="spoken2written"><input type="checkbox" id="spoken2written" /> 书面化（转写后 LLM 语句规范化）</label>
      </div>
    </div>
    <div class="toolbar">
      <button type="button" class="ghost" id="clearBtn">清空重来</button>
      <label class="file">选择音频<input id="file" type="file" accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg" /></label>
      <button type="button" class="warn" id="saveAsrErr">保存 ASR 错误</button>
      <button type="button" class="danger" id="saveEditErr">保存改写错误</button>
    </div>
    <div class="save-panel" id="savePanel" hidden>
      <h3 id="saveTitle">保存错误样例</h3>
      <p class="hint" id="saveHint">把系统输出当 hyp；在「正确文本」里填你期望的结果，方便后续训练。</p>
      <label class="field" for="saveHyp">系统输出（hyp）</label>
      <textarea id="saveHyp" readonly></textarea>
      <label class="field" for="saveGold">正确文本（gold）</label>
      <textarea id="saveGold" placeholder="填期望的正确结果…"></textarea>
      <label class="field" for="saveNote">备注（可选）</label>
      <input type="text" id="saveNote" placeholder="例如：提手旁听成铁手旁" maxlength="500" />
      <div class="save-actions">
        <button type="button" id="saveConfirm">确认保存</button>
        <button type="button" class="ghost" id="saveCancel">取消</button>
      </div>
    </div>
    <div class="drop" id="drop" style="margin-top:14px;padding:18px 14px">也可把音频拖到这里</div>
    <div class="samples">
      <p>官方样例</p>
      <div class="row" id="samples"></div>
    </div>
    <audio id="player" controls hidden></audio>
    <audio id="reply" controls hidden style="margin-top:14px"></audio>
    <p id="status"></p>
    <p id="err" hidden></p>
  </section>
  <p class="ready" id="ready">模型加载中…</p>
  <details class="logbox" id="logBox" open>
    <summary>调试日志（也写入 kimi/demo_app.log）</summary>
    <pre id="logView">等待操作…</pre>
    <div class="log-actions">
      <button type="button" class="ghost" id="logRefresh">拉取服务端日志</button>
      <button type="button" class="ghost" id="logClear">清空显示</button>
      <button type="button" class="ghost" id="logCopy">复制</button>
    </div>
  </details>
</main>
<script>
function logLine(...parts) {
  const stamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const msg = parts.map((p) => {
    if (p == null) return String(p);
    if (typeof p === "string") return p;
    try { return JSON.stringify(p); } catch (_) { return String(p); }
  }).join(" ");
  const line = "[" + stamp + "] " + msg;
  console.log(line);
  const el = document.getElementById("logView");
  if (!el) return;
  if (el.textContent === "等待操作…") el.textContent = "";
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

async function pullServerLogs() {
  try {
    const r = await fetch("/api/logs?n=80");
    const d = await r.json();
    logLine("—— 服务端日志 ——");
    (d.lines || []).forEach((ln) => {
      const el = document.getElementById("logView");
      if (el.textContent === "等待操作…") el.textContent = "";
      el.textContent += ln + "\n";
    });
    const el = document.getElementById("logView");
    el.scrollTop = el.scrollHeight;
  } catch (e) {
    logLine("拉取服务端日志失败", e.message || String(e));
  }
}

document.getElementById("logRefresh").addEventListener("click", () => pullServerLogs());
document.getElementById("logClear").addEventListener("click", () => {
  document.getElementById("logView").textContent = "等待操作…";
});
document.getElementById("logCopy").addEventListener("click", async () => {
  const t = document.getElementById("logView").textContent || "";
  try {
    await navigator.clipboard.writeText(t);
    logLine("已复制到剪贴板");
  } catch (_) {
    logLine("复制失败，请手动选择日志文本");
  }
});

const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const recBtn = document.getElementById("rec");
const player = document.getElementById("player");
const replyPlayer = document.getElementById("reply");
const statusEl = document.getElementById("status");
const errEl = document.getElementById("err");
const samplesEl = document.getElementById("samples");
const readyEl = document.getElementById("ready");
const stageTitle = document.getElementById("stageTitle");
const draftBox = document.getElementById("draftBox");
const metaEl = document.getElementById("meta");
const clearBtn = document.getElementById("clearBtn");
const micHint = document.getElementById("micHint");
const spoken2writtenEl = document.getElementById("spoken2written");
const cmpPanel = document.getElementById("cmpPanel");
const asrTextDbg = document.getElementById("asrTextDbg");
const writtenTextDbg = document.getElementById("writtenTextDbg");
const cmpMeta = document.getElementById("cmpMeta");
const saveAsrErr = document.getElementById("saveAsrErr");
const saveEditErr = document.getElementById("saveEditErr");

function showAsrWrittenCompare(data) {
  const asr = (data.asr_text != null ? data.asr_text : data.text || "").trim();
  const written = (data.written_text != null ? data.written_text : "").trim();
  const didS2w = !!(data.spoken2written && (data.spoken2written_via || written));
  if (!asr && !written) {
    cmpPanel.hidden = true;
    return;
  }
  cmpPanel.hidden = false;
  asrTextDbg.value = asr;
  if (didS2w) {
    writtenTextDbg.value = written || (data.text || "").trim();
  } else if (data.spoken2written) {
    writtenTextDbg.value = written || (data.text || "").trim() || "（书面化未改动 / 失败，与转写相同）";
  } else {
    writtenTextDbg.value = "（未开书面化）";
  }
  const bits = [];
  if (data.ms_asr != null) bits.push("ASR " + data.ms_asr + "ms");
  if (data.ms_s2w) bits.push("书面化 " + data.ms_s2w + "ms");
  if (data.spoken2written_via) bits.push(data.spoken2written_via);
  if (asr && written && asr === written) bits.push("两者相同");
  cmpMeta.textContent = bits.join(" · ") || "";
}
const S2W_KEY = "kimi_spoken2written";
try {
  const saved = localStorage.getItem(S2W_KEY);
  if (saved === "1") spoken2writtenEl.checked = true;
  if (saved === "0") spoken2writtenEl.checked = false;
} catch (_) {}
spoken2writtenEl.addEventListener("change", () => {
  try { localStorage.setItem(S2W_KEY, spoken2writtenEl.checked ? "1" : "0"); } catch (_) {}
  logLine("书面化", spoken2writtenEl.checked ? "开" : "关");
});
function spoken2writtenOn() { return !!spoken2writtenEl.checked; }
const savePanel = document.getElementById("savePanel");
const saveTitle = document.getElementById("saveTitle");
const saveHint = document.getElementById("saveHint");
const saveHyp = document.getElementById("saveHyp");
const saveGold = document.getElementById("saveGold");
const saveNote = document.getElementById("saveNote");
const saveConfirm = document.getElementById("saveConfirm");
const saveCancel = document.getElementById("saveCancel");

let mode = "asr";
let recorder = null;
let chunks = [];
let busy = false;
let recStream = null;
/** Last turn context for error-sample export. */
let lastCtx = {
  kind: null, // 'asr' | 'edit'
  hyp: "",
  draft_before: "",
  instruction: "",
  edit_via: "",
  clarify_via: "",
  normalize_via: "",
  asr_engine: "",
  ms: 0,
  audioBlob: null,
  audioName: "",
};
let pendingSaveType = null; // 'asr' | 'edit'

function blobToB64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => {
      const s = String(fr.result || "");
      const i = s.indexOf(",");
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    fr.onerror = () => reject(fr.error || new Error("read failed"));
    fr.readAsDataURL(blob);
  });
}

function closeSavePanel() {
  savePanel.hidden = true;
  pendingSaveType = null;
}

function openSavePanel(errorType) {
  if (!lastCtx.kind) {
    showError("还没有可保存的结果，请先录音转写或改写");
    return;
  }
  if (errorType === "edit" && lastCtx.kind !== "edit") {
    showError("当前最近一次是转写，不是改写。若要标改写错误，请先再说一条修改指令。");
    return;
  }
  if (errorType === "asr" && lastCtx.kind !== "asr") {
    // still allow: user may want to mark the ASR of the edit instruction, but primary is first-pass ASR
    // Prefer warning but allow saving last hyp as asr error if they insist? User asked to distinguish.
    // Strict: only when last was asr.
    showError("当前最近一次是改写。若要标 ASR 错误，请先清空后重新转写，或先保存改写错误。");
    return;
  }
  pendingSaveType = errorType;
  savePanel.hidden = false;
  if (errorType === "asr") {
    saveTitle.textContent = "保存 ASR 识别错误";
    saveHint.textContent = "单纯转写听错。hyp=系统转写；gold=你听到的正确文案。会连同最近一次录音一起存。";
  } else {
    saveTitle.textContent = "保存改写不符合要求";
    saveHint.textContent = "语音修改结果不对。hyp=改写结果；gold=期望改写。会保存改前草稿、识别到的指令与音频。";
  }
  saveHyp.value = lastCtx.hyp || currentDraft();
  saveGold.value = "";
  saveNote.value = "";
  saveGold.focus();
}

function currentDraft() {
  return (draftBox.value || "").trim();
}

function setDraft(text) {
  draftBox.value = text || "";
  refreshHint();
}

function refreshHint() {
  if (mode === "chat") {
    micHint.textContent = "点击开始，再点一次结束并回答";
    stageTitle.textContent = "语音问答";
    return;
  }
  stageTitle.textContent = "当前文本";
  micHint.textContent = currentDraft()
    ? "框里已有文字 → 再说就是修改指令"
    : "框是空的 → 先说一段做转写";
}

function pickMime() {
  const cands = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  for (const t of cands) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

function micBlockedReason(err) {
  if (!window.isSecureContext) {
    return "当前页面不是安全上下文，浏览器禁止麦克风。本机请用 http://localhost:7861/ ；局域网请用 https://本机IP:7862/（自签名点「高级 → 继续访问」）。";
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return "浏览器不支持麦克风录音，请换 Chrome / Edge。";
  }
  const name = (err && err.name) || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "麦克风权限被拒绝，请在浏览器地址栏允许麦克风后重试。";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "未检测到麦克风设备。";
  }
  return "无法打开麦克风：" + ((err && err.message) || String(err || "未知错误"));
}

async function openMic() {
  if (!window.isSecureContext) throw Object.assign(new Error("insecure"), { name: "SecurityError" });
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      channelCount: 1,
    },
  });
}

document.querySelectorAll(".modes button").forEach((b) => {
  b.addEventListener("click", () => {
    mode = b.dataset.mode;
    document.querySelectorAll(".modes button").forEach((x) => x.classList.toggle("on", x === b));
    refreshHint();
  });
});

draftBox.addEventListener("input", () => refreshHint());

clearBtn.addEventListener("click", () => {
  setDraft("");
  cmpPanel.hidden = true;
  asrTextDbg.value = "";
  writtenTextDbg.value = "";
  cmpMeta.textContent = "";
  metaEl.textContent = "";
  statusEl.textContent = "";
  showError("");
  player.hidden = true;
  replyPlayer.hidden = true;
  replyPlayer.removeAttribute("src");
  closeSavePanel();
  lastCtx = {
    kind: null, hyp: "", asr_text: "", written_text: "",
    draft_before: "", instruction: "", edit_via: "",
    clarify_via: "", normalize_via: "", asr_engine: "", ms: 0,
    audioBlob: null, audioName: "",
  };
  logLine("已清空，下一次说话将重新转写");
});

saveAsrErr.addEventListener("click", () => openSavePanel("asr"));
saveEditErr.addEventListener("click", () => openSavePanel("edit"));
saveCancel.addEventListener("click", () => closeSavePanel());
saveConfirm.addEventListener("click", async () => {
  if (!pendingSaveType || !lastCtx.kind) {
    showError("没有待保存的样例");
    return;
  }
  const gold = (saveGold.value || "").trim();
  if (!gold) {
    showError("请填写正确文本（gold）");
    return;
  }
  const hyp = (saveHyp.value || lastCtx.hyp || "").trim();
  setBusy(true, "正在保存错误样例…");
  try {
    let audio_b64 = "";
    if (lastCtx.audioBlob) {
      audio_b64 = await blobToB64(lastCtx.audioBlob);
    }
    const payload = {
      error_type: pendingSaveType,
      hyp,
      gold,
      note: (saveNote.value || "").trim(),
      draft_before: lastCtx.draft_before || "",
      instruction: lastCtx.instruction || "",
      edit_via: lastCtx.edit_via || "",
      clarify_via: lastCtx.clarify_via || "",
      normalize_via: lastCtx.normalize_via || "",
      asr_engine: lastCtx.asr_engine || "",
      ms: lastCtx.ms || 0,
      audio_b64,
      audio_name: lastCtx.audioName || "",
      audio_type: (lastCtx.audioBlob && lastCtx.audioBlob.type) || "",
    };
    const res = await fetch("/api/save-error", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "保存失败");
    logLine("已保存错误样例", data.error_type, data.id, data.path || "");
    statusEl.textContent = "已保存 · " + data.error_type + " · " + data.id;
    showError("");
    closeSavePanel();
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setBusy(false);
    refreshHint();
  }
});

function setBusy(on, text) {
  busy = on;
  const recLive = recorder && recorder.state === "recording";
  recBtn.disabled = on && !recLive;
  fileInput.disabled = on;
  clearBtn.disabled = on;
  draftBox.disabled = on;
  saveAsrErr.disabled = on;
  saveEditErr.disabled = on;
  saveConfirm.disabled = on;
  statusEl.textContent = text || "";
  for (const b of samplesEl.querySelectorAll("button")) b.disabled = on;
}

function showError(msg) {
  errEl.hidden = !msg;
  errEl.textContent = msg || "";
  if (msg) logLine("错误", msg);
}

function showAudio(blob) {
  player.hidden = false;
  player.src = URL.createObjectURL(blob);
}

function utf8ToB64(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  bytes.forEach((b) => { bin += String.fromCharCode(b); });
  return btoa(bin);
}

function applyResult(data, audioBlob, audioName) {
  const text = (data.text || "").trim();
  logLine("结果", data.label || mode, text || "（空）", data.ms ? (data.ms + "ms") : "");
  if (data.instruction) logLine("识别指令", data.instruction);
  if (data.edit_via) logLine("改写路径", data.edit_via);
  if (data.clarify_via) logLine("姓名纠字", data.clarify_via);
  if (data.normalize_via) logLine("列表规范化", data.normalize_via);
  if (data.filler_via) logLine("去填充词", data.filler_via);
  if (data.spoken2written_via) logLine("书面化", data.spoken2written_via);
  if (data.reply_wav) {
    replyPlayer.hidden = false;
    replyPlayer.src = data.reply_wav;
  } else {
    replyPlayer.hidden = true;
    replyPlayer.removeAttribute("src");
  }
  if (mode === "chat" || data.mode === "chat") {
    setDraft(text);
    metaEl.textContent = "问答 · " + (data.ms ? (data.ms / 1000).toFixed(1) + "s" : "");
    lastCtx.kind = null;
    return;
  }
  if (data.label === "改写" || data.mode === "edit") {
    const prev = (data.draft || currentDraft() || "").trim();
    setDraft(text || prev);
    const bits = ["已改写"];
    if (data.instruction) bits.push("指令：" + data.instruction);
    if (data.edit_via) bits.push(data.edit_via);
    if (prev && prev !== currentDraft()) bits.push("原稿：" + prev);
    metaEl.textContent = bits.join(" · ");
    lastCtx = {
      kind: "edit",
      hyp: (text || prev),
      draft_before: prev,
      instruction: data.instruction || "",
      edit_via: data.edit_via || "",
      clarify_via: data.clarify_via || "",
      normalize_via: data.normalize_via || "",
      asr_engine: data.asr_engine || lastCtx.asr_engine || "",
      ms: data.ms || 0,
      audioBlob: audioBlob || lastCtx.audioBlob,
      audioName: audioName || lastCtx.audioName || "",
    };
    return;
  }
  // ASR / 转写：当前文本用最终结果（书面化后覆盖）；对照面板同时展示转写原文
  setDraft(text);
  showAsrWrittenCompare(data);
  if (data.asr_text) logLine("转写原文", data.asr_text);
  if (data.written_text) logLine("书面化结果", data.written_text);
  const bits = ["已转写"];
  if (data.spoken2written && data.spoken2written_via) bits.push("已写入书面化");
  if (data.clarify_via) bits.push(data.clarify_via);
  if (data.normalize_via) bits.push(data.normalize_via);
  if (data.filler_via) bits.push(data.filler_via);
  if (data.spoken2written_via) bits.push(data.spoken2written_via);
  if (data.asr_engine) bits.push(data.asr_engine);
  metaEl.textContent = bits.join(" · ");
  lastCtx = {
    kind: "asr",
    hyp: text,
    asr_text: data.asr_text || text,
    written_text: data.written_text || "",
    draft_before: "",
    instruction: "",
    edit_via: "",
    clarify_via: data.clarify_via || "",
    normalize_via: data.normalize_via || "",
    filler_via: data.filler_via || "",
    spoken2written_via: data.spoken2written_via || "",
    asr_engine: data.asr_engine || "",
    ms: data.ms || 0,
    audioBlob: audioBlob || null,
    audioName: audioName || "",
  };
}

async function handleAudio(blob, name) {
  showError("");
  showAudio(blob);
  closeSavePanel();
  const draft = currentDraft();
  // 语音编辑模式：框里有字 → 当修改指令；否则转写
  const doEdit = mode === "asr" && !!draft;
  if (doEdit) {
    setBusy(true, "正在按语音修改…");
    logLine("改写", "draft=" + draft, "bytes=" + (blob.size || 0));
    try {
      const res = await fetch("/api/edit", {
        method: "POST",
        headers: {
          "Content-Type": blob.type || "application/octet-stream",
          "X-Filename": name || "edit.bin",
          "X-Draft-B64": utf8ToB64(draft),
        },
        body: blob,
      });
      const data = await res.json();
      if (!res.ok) {
        if (data.traceback) logLine("服务端堆栈\n" + data.traceback);
        throw new Error(data.error || "改写失败");
      }
      applyResult(data, blob, name || "edit.bin");
      statusEl.textContent = "改写完成 · " + (data.ms ? (data.ms / 1000).toFixed(1) + "s" : "");
      pullServerLogs();
    } catch (err) {
      showError(err.message || String(err));
      statusEl.textContent = "";
      pullServerLogs();
    } finally {
      setBusy(false);
      refreshHint();
    }
    return;
  }

  setBusy(true, mode === "chat" ? "正在回答…" : (spoken2writtenOn() ? "正在转写并书面化…" : "正在转写…"));
  logLine("转写/问答", mode, name || "audio.bin", "bytes=" + (blob.size || 0), "书面化=" + (spoken2writtenOn() ? "1" : "0"));
  try {
    const qs = "mode=" + encodeURIComponent(mode)
      + "&spoken2written=" + (spoken2writtenOn() ? "1" : "0");
    const res = await fetch("/api/infer?" + qs, {
      method: "POST",
      headers: {
        "Content-Type": blob.type || "application/octet-stream",
        "X-Filename": name || "audio.bin",
        "X-Spoken2Written": spoken2writtenOn() ? "1" : "0",
      },
      body: blob,
    });
    const data = await res.json();
    if (!res.ok) {
      if (data.traceback) logLine("服务端堆栈\n" + data.traceback);
      throw new Error(data.error || "失败");
    }
    applyResult(data, blob, name || "audio.bin");
    statusEl.textContent = "完成 · " + (data.ms ? (data.ms / 1000).toFixed(1) + "s" : "");
    pullServerLogs();
  } catch (err) {
    showError(err.message || String(err));
    statusEl.textContent = "";
    pullServerLogs();
  } finally {
    setBusy(false);
    refreshHint();
  }
}

async function inferSample(id) {
  showError("");
  player.hidden = true;
  setBusy(true, "跑样例…");
  try {
    const res = await fetch("/api/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "失败");
    if (data.sample_wav) {
      player.hidden = false;
      player.src = data.sample_wav;
    }
    mode = data.mode || mode;
    document.querySelectorAll(".modes button").forEach((x) => x.classList.toggle("on", x.dataset.mode === mode));
    applyResult(data);
    statusEl.textContent = "样例完成";
  } catch (err) {
    showError(err.message || String(err));
    statusEl.textContent = "";
  } finally {
    setBusy(false);
    refreshHint();
  }
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  if (f) handleAudio(f, f.name);
});
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) handleAudio(f, f.name);
});

async function toggleMainRec() {
  showError("");
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  if (busy) return;
  try {
    recStream = await openMic();
    chunks = [];
    const mime = pickMime();
    recorder = mime ? new MediaRecorder(recStream, { mimeType: mime }) : new MediaRecorder(recStream);
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      if (recStream) {
        recStream.getTracks().forEach((t) => t.stop());
        recStream = null;
      }
      recBtn.classList.remove("live");
      recBtn.textContent = "按住说";
      refreshHint();
      const type = (recorder && recorder.mimeType) || mime || "audio/webm";
      const blob = new Blob(chunks, { type });
      const ext = type.includes("mp4") ? "m4a" : (type.includes("ogg") ? "ogg" : "webm");
      handleAudio(blob, "recording." + ext);
    };
    recorder.start(250);
    recBtn.classList.add("live");
    recBtn.textContent = "录音中";
    micHint.textContent = currentDraft() && mode === "asr"
      ? "说完再点一次 → 按语音修改"
      : "说完再点一次 → 结束并识别";
    statusEl.textContent = "正在录音…对准麦克风说话";
  } catch (err) {
    showError(micBlockedReason(err));
  }
}
recBtn.addEventListener("click", () => toggleMainRec());

if (!window.isSecureContext) {
  showError(micBlockedReason(null));
  micHint.textContent = "需用 HTTPS 才能用麦克风";
}

fetch("/api/samples").then((r) => r.json()).then((rows) => {
  for (const row of rows) {
    const b = document.createElement("button");
    b.className = "ghost";
    b.type = "button";
    b.textContent = row.label;
    b.addEventListener("click", () => inferSample(row.id));
    samplesEl.appendChild(b);
  }
});

async function pollReady() {
  try {
    const r = await fetch("/api/ready");
    const d = await r.json();
    if (d.ready) {
      try {
        if (localStorage.getItem(S2W_KEY) == null && d.spoken2written_default) {
          spoken2writtenEl.checked = true;
        }
      } catch (_) {}
      const llm = d.llm_ready ? "LLM已加载" : "LLM按需加载";
      const s2w = spoken2writtenOn() || d.spoken2written_default ? " · 书面化可开" : "";
      readyEl.textContent = "SenseVoice 就绪 · " + llm + (d.speech ? " · 含语音回复" : "") + s2w;
      readyEl.className = "ready ok";
      logLine("SenseVoice 就绪", llm, "书面化默认=" + (d.spoken2written_default ? "开" : "关"));
      refreshHint();
      return;
    }
    readyEl.textContent = d.error ? ("加载失败：" + d.error) : "SenseVoice 加载中…";
    readyEl.className = d.error ? "ready bad" : "ready";
    if (!d.error) setTimeout(pollReady, 2000);
  } catch (e) {
    readyEl.textContent = "服务未响应";
    readyEl.className = "ready bad";
    setTimeout(pollReady, 3000);
  }
}
refreshHint();
pollReady();
</script>
</body>
</html>
"""


def _to_wav16k(src: Path) -> Path:
    """Return a 16 kHz mono wav path (temp file caller must delete if different from src)."""
    if src.suffix.lower() == ".wav":
        try:
            with wave.open(str(src), "rb") as w:
                if w.getframerate() == 16000 and w.getnchannels() == 1:
                    return src
        except wave.Error:
            pass
    dst = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst


def _write_wav_bytes(path: Path, audio: np.ndarray, sr: int = 24000) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _decode_draft_b64(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        return base64.b64decode(raw.encode("ascii")).decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"X-Draft-B64 无效: {exc}") from exc


def _save_error_sample(payload: dict) -> dict:
    """Persist a labeled error case for later training/eval."""
    error_type = (payload.get("error_type") or "").strip()
    if error_type not in ("asr", "edit"):
        raise ValueError("error_type 必须是 asr 或 edit")
    hyp = (payload.get("hyp") or "").strip()
    gold = (payload.get("gold") or "").strip()
    if not gold:
        raise ValueError("缺少正确文本 gold")
    if not hyp:
        raise ValueError("缺少系统输出 hyp")
    note = (payload.get("note") or "").strip()
    if len(note) > MAX_NOTE_CHARS:
        raise ValueError(f"备注过长（>{MAX_NOTE_CHARS}）")
    if len(gold) > MAX_DRAFT_CHARS or len(hyp) > MAX_DRAFT_CHARS:
        raise ValueError(f"文本过长（>{MAX_DRAFT_CHARS}）")

    ERROR_SAMPLES.mkdir(parents=True, exist_ok=True)
    ERROR_WAVS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    # short unique suffix
    import secrets

    sid = f"{error_type}_{stamp}_{secrets.token_hex(3)}"
    wav_rel = ""
    audio_b64 = payload.get("audio_b64") or ""
    audio_name = payload.get("audio_name") or ""
    if audio_b64:
        try:
            raw = base64.b64decode(audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"audio_b64 无效: {exc}") from exc
        if len(raw) > MAX_BYTES:
            raise ValueError("音频太大")
        suffix = Path(audio_name).suffix.lower() if audio_name else ""
        if suffix not in (".wav", ".webm", ".ogg", ".m4a", ".mp3", ".bin"):
            ctype = (payload.get("audio_type") or "").lower()
            if "wav" in ctype:
                suffix = ".wav"
            elif "ogg" in ctype:
                suffix = ".ogg"
            elif "mp4" in ctype or "m4a" in ctype:
                suffix = ".m4a"
            else:
                suffix = ".webm"
        wav_path = ERROR_WAVS / f"{sid}{suffix}"
        wav_path.write_bytes(raw)
        wav_rel = str(wav_path.relative_to(ROOT))

    row = {
        "id": sid,
        "error_type": error_type,  # asr = 纯识别错；edit = 改写不符合
        "hyp": hyp,
        "gold": gold,
        "note": note,
        "draft_before": (payload.get("draft_before") or "").strip(),
        "instruction": (payload.get("instruction") or "").strip(),
        "edit_via": (payload.get("edit_via") or "").strip(),
        "clarify_via": (payload.get("clarify_via") or "").strip(),
        "normalize_via": (payload.get("normalize_via") or "").strip(),
        "asr_engine": (payload.get("asr_engine") or "").strip(),
        "ms": int(payload.get("ms") or 0),
        "wav": wav_rel,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    with ERROR_LOCK:
        with ERROR_CASES.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _slog(
        "saved error sample id=%s type=%s hyp=%r gold=%r wav=%s",
        sid,
        error_type,
        hyp[:80],
        gold[:80],
        wav_rel or "-",
    )
    return {
        "id": sid,
        "error_type": error_type,
        "path": str(ERROR_CASES.relative_to(ROOT)),
        "wav": wav_rel,
    }


class Demo:
    def __init__(
        self,
        model_path: Path,
        *,
        speech: bool,
        sensevoice_path: Path,
        sensevoice_device: str = "cuda:0",
        spoken2written_default: bool = False,
    ):
        self.model_path = model_path
        self.speech = speech
        self.spoken2written_default = bool(spoken2written_default)
        self.asr = SenseVoiceASR(sensevoice_path, device=sensevoice_device)
        self.model = None  # Kimi-Audio, lazy
        self.lock = threading.Lock()  # Kimi generate lock
        self.llm_ready = False
        self.llm_loading = False
        self.load_error: str | None = None
        self._reply_dir = Path(tempfile.mkdtemp(prefix="kimi_demo_"))

    @property
    def ready(self) -> bool:
        return self.asr.ready

    def load(self) -> None:
        """Startup: SenseVoice only. Kimi loads on first LLM rewrite / chat."""
        try:
            print(
                f"loading SenseVoice from {self.asr.model_dir} device={self.asr.device}",
                flush=True,
            )
            _slog(
                "loading SenseVoice path=%s device=%s",
                self.asr.model_dir,
                self.asr.device,
            )
            self.asr.load()
            print("sensevoice asr ready", flush=True)
            _slog("sensevoice ready")
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            print(f"sensevoice load failed: {self.load_error}", flush=True)
            _slog("sensevoice load failed: %s", self.load_error)

    def ensure_llm(self) -> None:
        if self.llm_ready and self.model is not None:
            return
        with self.lock:
            if self.llm_ready and self.model is not None:
                return
            self.llm_loading = True
            try:
                from kimia_infer.api.kimia import KimiAudio

                print(
                    f"loading KimiAudio (lazy) from {self.model_path} speech={self.speech}",
                    flush=True,
                )
                _slog(
                    "loading KimiAudio lazy path=%s speech=%s",
                    self.model_path,
                    self.speech,
                )
                self.model = KimiAudio(
                    model_path=str(self.model_path),
                    load_detokenizer=self.speech,
                )
                self.llm_ready = True
                print("kimi llm ready", flush=True)
                _slog("kimi llm ready")
            except Exception as exc:  # noqa: BLE001
                self.load_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                _slog("kimi llm load failed: %s", self.load_error)
                raise RuntimeError(self.load_error) from exc
            finally:
                self.llm_loading = False

    def _asr(self, wav_path: Path) -> str:
        if not self.asr.ready:
            raise RuntimeError(self.load_error or self.asr.load_error or "SenseVoice 还在加载")
        return _normalize_display(self.asr.transcribe(wav_path))

    def _clarify_name_text(self, text: str) -> tuple[str, str | None]:
        """Post-ASR name glyph clarify: rules first, LLM if pattern but unresolved."""
        raw = (text or "").strip()
        if not raw:
            return raw, None
        ruled = _resolve_name_clarify(raw)
        if ruled:
            fixed, via = ruled
            _slog("name clarify via=%s %r -> %r", via, raw, fixed)
            return fixed, via
        if not _has_name_clarify(raw):
            return raw, None
        try:
            self.ensure_llm()
            assert self.model is not None
            messages = [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": NAME_CLARIFY_PROMPT.format(text=raw),
                },
            ]
            clarify_sampling = {**EDIT_SAMPLING, "max_new_tokens": 96}
            with self.lock:
                _, out = self.model.generate(
                    messages, **clarify_sampling, output_type="text"
                )
            cleaned = _clean_rewrite((out or "").strip(), draft=raw)
            cleaned = _normalize_display(cleaned)
            # keep a single sentence if the model loops
            if cleaned.count("我叫") > 1 or cleaned.count("我是") > 1:
                for sep in ("。", "！", "？", ".", "!", "?"):
                    if sep in cleaned:
                        cleaned = cleaned.split(sep, 1)[0] + sep
                        break
            if cleaned and not _looks_like_garbage(cleaned, raw):
                _slog("name clarify via=llm %r -> %r", raw, cleaned)
                return cleaned, "llm_name_clarify"
            _slog("name clarify llm rejected, keep raw")
        except Exception as exc:  # noqa: BLE001
            _slog("name clarify llm failed: %s: %s", type(exc).__name__, exc)
            traceback.print_exc()
        return raw, None

    def _normalize_list_text(self, text: str) -> tuple[str, str | None]:
        """Post-ASR list structure normalize (rules only)."""
        raw = (text or "").strip()
        if not raw:
            return raw, None
        ruled = _normalize_list_structure(raw)
        if not ruled:
            return raw, None
        fixed, via = ruled
        _slog("list normalize via=%s %r -> %r", via, raw, fixed)
        return fixed, via

    def _strip_filler_text(self, text: str) -> tuple[str, str | None]:
        raw = (text or "").strip()
        if not raw:
            return raw, None
        ruled = _strip_fillers(raw)
        if not ruled:
            return raw, None
        fixed, via = ruled
        _slog("filler via=%s %r -> %r", via, raw, fixed)
        return fixed, via

    def _spoken2written_llm(self, text: str) -> tuple[str, str | None]:
        """LLM spoken→written normalize. Returns (text, via)."""
        raw = (text or "").strip()
        if not raw:
            return raw, None
        try:
            self.ensure_llm()
            assert self.model is not None
            messages = [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": SPOKEN2WRITTEN_PROMPT.format(text=raw),
                },
            ]
            with self.lock:
                _, out = self.model.generate(
                    messages, **S2W_SAMPLING, output_type="text"
                )
            raw_out = (out or "").strip()
            cleaned = _clean_spoken2written(raw_out, draft=raw)
            cleaned = _normalize_display(cleaned)
            if cleaned and not _looks_like_bad_spoken2written(cleaned, raw):
                _slog(
                    "spoken2written via=llm raw=%r -> %r",
                    raw_out[:160],
                    cleaned,
                )
                return cleaned, "llm_spoken2written"
            _slog(
                "spoken2written llm rejected raw=%r cleaned=%r, keep asr",
                raw_out[:160],
                cleaned,
            )
        except Exception as exc:  # noqa: BLE001
            _slog("spoken2written llm failed: %s: %s", type(exc).__name__, exc)
            traceback.print_exc()
        return raw, None

    def run(
        self,
        wav_path: Path,
        mode: str,
        *,
        spoken2written: bool | None = None,
    ) -> dict:
        import time

        t0 = time.time()
        do_s2w = (
            self.spoken2written_default
            if spoken2written is None
            else bool(spoken2written)
        )
        converted = _to_wav16k(wav_path)
        try:
            if mode == "asr":
                text = self._asr(converted)
                t_asr = time.time()
                text, clarify_via = self._clarify_name_text(text)
                text, normalize_via = self._normalize_list_text(text)
                text, filler_via = self._strip_filler_text(text)
                asr_text = text  # after rules, before spoken2written LLM
                spoken2written_via = None
                written_text = None
                t_pre_s2w = time.time()
                if do_s2w:
                    written, spoken2written_via = self._spoken2written_llm(asr_text)
                    written_text = written
                    text = written  # draft / primary text = 书面化结果
                t_end = time.time()
                out: dict = {
                    "label": "转写",
                    "text": text,
                    "asr_text": asr_text,
                    "mode": mode,
                    "asr_engine": "sensevoice",
                    "spoken2written": do_s2w,
                    "ms": int((t_end - t0) * 1000),
                    "ms_asr": int((t_asr - t0) * 1000),
                    "ms_s2w": int((t_end - t_pre_s2w) * 1000) if do_s2w else 0,
                }
                if do_s2w:
                    out["written_text"] = written_text if written_text is not None else text
                if clarify_via:
                    out["clarify_via"] = clarify_via
                if normalize_via:
                    out["normalize_via"] = normalize_via
                if filler_via:
                    out["filler_via"] = filler_via
                if spoken2written_via:
                    out["spoken2written_via"] = spoken2written_via
                _slog(
                    "转写 done engine=sensevoice clarify=%s normalize=%s filler=%s s2w=%s via=%s asr=%r written=%r ms=%s (asr=%s s2w=%s)",
                    clarify_via or "-",
                    normalize_via or "-",
                    filler_via or "-",
                    do_s2w,
                    spoken2written_via or "-",
                    asr_text,
                    out.get("written_text") or "-",
                    out["ms"],
                    out["ms_asr"],
                    out["ms_s2w"],
                )
                return out

            # chat still needs Kimi audio understanding
            self.ensure_llm()
            assert self.model is not None
            messages = [
                {"role": "user", "message_type": "audio", "content": str(converted)},
            ]
            output_type = "both" if self.speech else "text"
            with self.lock:
                wav_out, text = self.model.generate(
                    messages, **SAMPLING, output_type=output_type
                )
            out = {
                "label": "问答",
                "text": _normalize_display((text or "").strip()),
                "mode": mode,
                "ms": int((time.time() - t0) * 1000),
            }
            _slog("问答 done text=%r ms=%s", out["text"], out["ms"])
            if self.speech and wav_out is not None and output_type == "both":
                reply = self._reply_dir / f"reply_{int(t0 * 1000)}.wav"
                arr = wav_out[0] if isinstance(wav_out, (list, tuple)) else wav_out
                if hasattr(arr, "cpu"):
                    arr = arr.detach().cpu().numpy()
                _write_wav_bytes(reply, np.asarray(arr).reshape(-1), sr=24000)
                out["reply_wav"] = f"/api/reply/{reply.name}"
            return out
        finally:
            if converted != wav_path:
                converted.unlink(missing_ok=True)

    def run_edit(self, draft_text: str, edit_wav: Path) -> dict:
        if not self.asr.ready:
            raise RuntimeError(self.load_error or self.asr.load_error or "SenseVoice 还在加载")
        draft = _normalize_display((draft_text or "").strip())
        if not draft:
            raise ValueError("缺少转写草稿")
        if len(draft) > MAX_DRAFT_CHARS:
            raise ValueError(f"草稿过长（>{MAX_DRAFT_CHARS} 字）")
        import time

        t0 = time.time()
        _slog("edit start draft=%r wav=%s size=%s", draft, edit_wav, edit_wav.stat().st_size)
        converted = _to_wav16k(edit_wav)
        try:
            instruction = self._asr(converted)
            _slog(
                "edit asr engine=sensevoice instruction=%r (%.0fms)",
                instruction,
                (time.time() - t0) * 1000,
            )
            if not instruction:
                raise ValueError("未能识别修改指令，请重录")

            via = "llm"
            # Voice command: 书面化一下 → dedicated spoken→written LLM
            if _is_spoken2written_instruction(instruction):
                text, s2w_via = self._spoken2written_llm(draft)
                via = s2w_via or "llm_spoken2written+keep"
                text = text or draft
                _slog("edit spoken2written via=%s -> %r", via, text)
            else:
                ruled = _apply_edit_rules(draft, instruction)
                if ruled:
                    text, via = ruled
                    _slog("edit rule via=%s -> %r", via, text)
                else:
                    rewrite_messages = [
                        {
                            "role": "user",
                            "message_type": "text",
                            "content": EDIT_REWRITE_PROMPT.format(
                                draft=draft, instruction=instruction
                            ),
                        },
                    ]
                    try:
                        self.ensure_llm()
                        assert self.model is not None
                        with self.lock:
                            _, text = self.model.generate(
                                rewrite_messages, **EDIT_SAMPLING, output_type="text"
                            )
                        raw = text or ""
                        text = _clean_rewrite(raw, draft=draft)
                        _slog(
                            "edit llm raw_len=%s cleaned=%r",
                            len(raw),
                            text[:120],
                        )
                        if _looks_like_garbage(text, draft):
                            _slog("edit llm cleaned still garbage, drop")
                            text = ""
                    except Exception as exc:  # noqa: BLE001
                        _slog("edit llm failed: %s: %s", type(exc).__name__, exc)
                        traceback.print_exc()
                        text = ""
                        via = "llm_fail"
                    if not text or text == instruction or text == draft:
                        fallback = (
                            _apply_radical_describe_replace(draft, instruction)
                            or _apply_toggle(draft, instruction)
                            or _apply_simple_replace(draft, instruction)
                            or _apply_fuzzy_replace(draft, instruction)
                        )
                        if fallback:
                            text = fallback
                            via = via + "+rule_fallback"
                            _slog("edit fallback rule -> %r", text)
                        else:
                            text = text or draft
                            via = via + "+keep"
                            _slog("edit keep draft/text -> %r", text)
            # never return runaway special-token spam to the page
            if via != "noop":
                text = _clean_rewrite(text, draft=draft) or draft
                if _looks_like_garbage(text, draft):
                    text = (
                        _apply_radical_describe_replace(draft, instruction)
                        or _apply_toggle(draft, instruction)
                        or _apply_simple_replace(draft, instruction)
                        or _apply_fuzzy_replace(draft, instruction)
                        or draft
                    )
                    via = via + "+sanitize"
            out = {
                "label": "改写",
                "text": _normalize_display(text),
                "draft": draft,
                "instruction": instruction,
                "edit_via": via,
                "asr_engine": "sensevoice",
                "mode": "edit",
                "ms": int((time.time() - t0) * 1000),
            }
            _slog("edit done via=%s text=%r ms=%s", via, text, out["ms"])
            return out
        finally:
            if converted != edit_wav:
                converted.unlink(missing_ok=True)


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _bytes(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_audio_upload(self) -> tuple[Path, str]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BYTES:
                raise ValueError("音频太大或为空")
            data = self.rfile.read(n)
            name = self.headers.get("X-Filename") or "upload.bin"
            suffix = Path(name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                return Path(tmp.name), name

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                raw = PAGE.encode("utf-8")
                self._bytes(200, raw, "text/html; charset=utf-8")
                return
            if self.path == "/api/ready":
                self._json(
                    200,
                    {
                        "ready": demo.ready,
                        "asr_engine": "sensevoice",
                        "llm_ready": demo.llm_ready,
                        "llm_loading": demo.llm_loading,
                        "speech": demo.speech,
                        "spoken2written_default": demo.spoken2written_default,
                        "error": demo.load_error or demo.asr.load_error,
                    },
                )
                return
            if self.path.startswith("/api/logs"):
                qs = parse_qs(urlparse(self.path).query)
                try:
                    n = max(1, min(400, int((qs.get("n") or ["100"])[0])))
                except ValueError:
                    n = 100
                with LOG_LOCK:
                    lines = list(LOG_BUF)[-n:]
                self._json(
                    200,
                    {
                        "lines": lines,
                        "file": str(LOG_FILE),
                    },
                )
                return
            if self.path == "/api/samples":
                self._json(200, [{"id": s["id"], "label": s["label"]} for s in SAMPLES])
                return
            if self.path.startswith("/api/reply/"):
                name = Path(self.path).name
                path = demo._reply_dir / name
                if not path.is_file():
                    self._json(404, {"error": "not found"})
                    return
                self._bytes(200, path.read_bytes(), "audio/wav")
                return
            if self.path.startswith("/api/sample-wav/"):
                sid = Path(self.path).name
                row = next((s for s in SAMPLES if s["id"] == sid), None)
                if not row:
                    self._json(404, {"error": "unknown sample"})
                    return
                path = TEST_AUDIOS / row["path"]
                self._bytes(200, path.read_bytes(), "audio/wav")
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path.startswith("/api/infer"):
                    qs = parse_qs(urlparse(self.path).query)
                    mode = (qs.get("mode") or ["asr"])[0]
                    if mode not in ("asr", "chat"):
                        mode = "asr"
                    # Request flag overrides server default; header also accepted.
                    req_s2w = _parse_truthy((qs.get("spoken2written") or [None])[0])
                    if req_s2w is None:
                        req_s2w = _parse_truthy(self.headers.get("X-Spoken2Written"))
                    src, _name = self._read_audio_upload()
                    try:
                        out = demo.run(src, mode, spoken2written=req_s2w)
                    finally:
                        src.unlink(missing_ok=True)
                    self._json(200, out)
                    return
                if self.path == "/api/edit":
                    try:
                        draft = _decode_draft_b64(self.headers.get("X-Draft-B64"))
                    except ValueError as exc:
                        self._json(400, {"error": str(exc)})
                        return
                    if not draft:
                        self._json(400, {"error": "缺少转写草稿"})
                        return
                    src, _name = self._read_audio_upload()
                    try:
                        out = demo.run_edit(draft, src)
                    finally:
                        src.unlink(missing_ok=True)
                    self._json(200, out)
                    return
                if self.path == "/api/sample":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n).decode("utf-8") if n else "{}")
                    sid = body.get("id")
                    row = next((s for s in SAMPLES if s["id"] == sid), None)
                    if not row:
                        self._json(400, {"error": "unknown sample"})
                        return
                    path = TEST_AUDIOS / row["path"]
                    out = demo.run(path, row["mode"])
                    out["sample_wav"] = f"/api/sample-wav/{row['id']}"
                    self._json(200, out)
                    return
                if self.path == "/api/save-error":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n <= 0 or n > MAX_BYTES * 2:
                        self._json(400, {"error": "请求体无效或过大"})
                        return
                    body = json.loads(self.rfile.read(n).decode("utf-8"))
                    out = _save_error_sample(body if isinstance(body, dict) else {})
                    self._json(200, out)
                    return
                self._json(404, {"error": "not found"})
            except ValueError as exc:
                _slog("bad request: %s", exc)
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                _slog("handler error: %s: %s", type(exc).__name__, exc)
                print(tb, flush=True)
                self._json(
                    500,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": tb,
                    },
                )

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_DEFAULT))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument(
        "--speech",
        action="store_true",
        help="Load audio detokenizer for spoken replies (heavy VRAM).",
    )
    ap.add_argument(
        "--sensevoice",
        default=str(DEFAULT_SENSEVOICE),
        help="SenseVoice-Small model dir (FunASR).",
    )
    ap.add_argument(
        "--sensevoice-device",
        default="cuda:0",
        help="Device for SenseVoice (default cuda:0).",
    )
    ap.add_argument(
        "--spoken2written",
        action="store_true",
        default=_env_truthy("KIMI_SPOKEN2WRITTEN", False),
        help="Default-on: after ASR, run LLM spoken→written normalize "
        "(env KIMI_SPOKEN2WRITTEN=1). Per-request ?spoken2written=0/1 overrides.",
    )
    ap.add_argument(
        "--https",
        action="store_true",
        help="Also serve HTTPS (needed for browser mic on non-localhost IP).",
    )
    ap.add_argument(
        "--https-port",
        type=int,
        default=0,
        help="HTTPS port when --https is set (default: HTTP port + 1).",
    )
    ap.add_argument(
        "--https-only",
        action="store_true",
        help="Serve only HTTPS on --port (no plain HTTP).",
    )
    ap.add_argument(
        "--cert",
        default=str(ROOT / "certs" / "cert.pem"),
        help="TLS certificate path (with --https / --https-only).",
    )
    ap.add_argument(
        "--key",
        default=str(ROOT / "certs" / "key.pem"),
        help="TLS private key path (with --https / --https-only).",
    )
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_dir():
        print(f"missing model dir: {model_path}", flush=True)
        return 1
    sensevoice_path = Path(args.sensevoice)
    if not sensevoice_path.is_dir():
        print(f"missing SenseVoice dir: {sensevoice_path}", flush=True)
        return 1

    demo = Demo(
        model_path,
        speech=args.speech,
        sensevoice_path=sensevoice_path,
        sensevoice_device=args.sensevoice_device,
        spoken2written_default=bool(args.spoken2written),
    )
    if demo.spoken2written_default:
        print("spoken2written default ON (ASR → LLM normalize)", flush=True)
    threading.Thread(target=demo.load, daemon=True).start()

    handler = make_handler(demo)
    want_tls = args.https or args.https_only
    cert = Path(args.cert)
    key = Path(args.key)
    ctx = None
    if want_tls:
        import ssl

        if not cert.is_file() or not key.is_file():
            print(f"missing TLS files: {cert} / {key}", flush=True)
            return 1
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert), str(key))

    if args.https_only:
        server = ThreadingHTTPServer((args.host, args.port), handler)
        assert ctx is not None
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        print(
            f"demo https://{args.host}:{args.port}/  asr=SenseVoice llm=lazy speech={args.speech}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("bye", flush=True)
        return 0

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"demo http://{args.host}:{args.port}/  asr=SenseVoice llm=lazy speech={args.speech}",
        flush=True,
    )
    if args.https:
        https_port = args.https_port or (args.port + 1)
        tls = ThreadingHTTPServer((args.host, https_port), handler)
        assert ctx is not None
        tls.socket = ctx.wrap_socket(tls.socket, server_side=True)
        threading.Thread(target=tls.serve_forever, daemon=True).start()
        print(
            f"demo https://{args.host}:{https_port}/  (LAN mic)",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
