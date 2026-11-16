# Kimi-Audio 本地部署

代码与权重目录：`/cmy/cmy/ranAudio/kimi`

## 环境

从现有 `qwen3` 环境复制得到 `kimi-audio`（`conda clone` 因本机 CUDA 包损坏失败，改为目录复制并修正 shebang）。

```bash
/cmy/cmy/miniconda3/envs/kimi-audio/bin/python ...
```

- Python 3.10 / torch 2.5.1+cu124 / flash_attn 2.8.3
- 源码：jsDelivr 镜像（GitHub 直连不通）
- 子模块 GLM-4-Voice：同样走 jsDelivr → `kimia_infer/models/tokenizer/glm4/`
- 权重：ModelScope `moonshotai/Kimi-Audio-7B-Instruct`（HF 镜像大文件易超时）
- 本地权重：`models/Kimi-Audio-7B-Instruct/`（约 35 片 LLM + whisper + vocoder + 可选 detokenizer 17.7G）

续下权重（推荐，aria2 并发）：

```bash
# 8 个文件并行，每文件 16 连接；缺啥补啥
/cmy/cmy/miniconda3/envs/kimi-audio/bin/python scripts/download_concurrent.py --jobs 8 --connections 16
# 日志：download_concurrent.log
```

或 ModelScope 单线程：

```bash
bash scripts/download_model.sh
```

## 能力总览

已实现的语义转写 / 语音修正类型见 **[CAPABILITIES.md](./CAPABILITIES.md)**（含书面化参数、去填充词、替换、toggle、偏旁、姓名纠字、列表规范化等）。

## 打开方式（必读）

**用 Cursor Ports 转发后打开 `http://localhost:7861`**，不要用 Pod IP。详见 **[OPEN.md](./OPEN.md)**。

```bash
bash /cmy/cmy/ranAudio/switch_asr_kimi.sh kimi          # 晚上：停 ASR + 启 Kimi + 自动 open
bash /cmy/cmy/ranAudio/kimi/scripts/restart_and_open.sh # 仅重启并打开
```

## 体验页

两段式主流程：**SenseVoice 转写 → 语音修改指令（SenseVoice）→ 规则 / Kimi LLM 改写**（另有「语音问答」，问答会懒加载 Kimi）。

- ASR：本地 `SenseVoiceSmall`（FunASR），默认路径可用环境变量 `KIMI_SENSEVOICE` 覆盖
- 转写后：姓名纠字 → 列表结构 → 去填充词；若 **书面化开** 再 LLM 口语→书面
- LLM：规则改写失败、姓名纠字兜底、书面化、或「语音问答」时懒加载 Kimi-Audio

```bash
# 默认 HTTP:7861（本机 localhost 可录音）+ HTTPS:7862（局域网 IP 麦克风）
# 勿与 S1.5 / 其它训练共卡（会 OOM）；等 GPU 空闲再启
nohup setsid bash scripts/run_demo.sh >> demo.log 2>&1 < /dev/null &
# 书面化默认开：KIMI_SPOKEN2WRITTEN=1 bash scripts/run_demo.sh
# 推荐打开：见 OPEN.md（Cursor Ports → http://localhost:7861）
# 只要 HTTP：KIMI_DEMO_HTTP_ONLY=1 ；只要 HTTPS：KIMI_DEMO_HTTPS_ONLY=1
```

麦克风：页面有大圆形录音按钮，点一下开始、再点结束并识别。`localhost` 的 HTTP 可用麦；用局域网 IP 时请走 HTTPS:7862。页面可勾选「书面化」。

环境变量：

- `HF_ENDPOINT=https://hf-mirror.com`（脚本已默认）
- `KIMI_GLM_TOKENIZER` 指向本地 `models/glm-4-voice-tokenizer`（已用镜像下好）
- `KIMI_DEMO_SPEECH=1` 才加载语音回复（显存很重）
- `KIMI_SPOKEN2WRITTEN=1` 转写后默认 LLM 语句规范化（也可页面勾选 / `?spoken2written=1`）

改写接口：`POST /api/edit`，音频 body + header `X-Draft-B64`（UTF-8 base64 的转写草稿）。

## 语音修改评测

测试集与报告在 [`eval_voice_edit/`](eval_voice_edit/)（`cases.jsonl` ~100 条，字段见 `schema.md`）。

```bash
cd /cmy/cmy/ranAudio/kimi
# 1) 保证 demo 已就绪：http://127.0.0.1:7861/api/ready
# 2) 合成指令音频（默认只要 edge；多引擎：VOICE_EDIT_ENGINES=edge,qwen_custom,qwen_design）
# 3) 规则 oracle + HTTP 端到端
bash scripts/run_voice_edit_eval.sh
# 报告：eval_voice_edit/reports/latest.md
```

分步：

```bash
# 只合成 edge
python scripts/gen_voice_edit_tts.py --engines edge
# 有 qwen3-tts 环境时再补多样性
QWEN_PY=/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python \
  python scripts/gen_voice_edit_tts.py --engines qwen_custom,qwen_design

# 纯文本规则回归（不占 GPU）
python scripts/eval_voice_edit.py --mode oracle-text

# 端到端（需 demo）
python scripts/eval_voice_edit.py --mode http --base http://127.0.0.1:7861 --engines edge
```

指标：`instr_cer`（指令 ASR）、`edit_exact` / `edit_cer`、`echo_fail`、`over_edit`；按 `bucket` 与 `engine` 分组。人工录音可放 `eval_voice_edit/wavs_human/{id}.wav`。

## 快速推理（ASR）


```bash
cd /cmy/cmy/ranAudio/kimi
/cmy/cmy/miniconda3/envs/kimi-audio/bin/python scripts/smoke_asr.py
```

默认读 `test_audios/asr_example.wav`，只出文本（`load_detokenizer=False`）。GPU 被其它任务占用时不要强行加载。

## 说明

- 依赖见 `requirements.txt`；本机安装用的是去掉 torch/flash_attn/deepspeed 钉死版本的 `requirements.runtime.txt`。
- 包通过 `site-packages/kimi_audio_repo.pth` 指向本目录，无需 editable 编译。
