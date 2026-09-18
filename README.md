# ranAudio

> 日期：2026-09-12  
> 独立仓库，与 [`MiniCPM-o-Demo`](../MiniCPM-o-Demo) **平级**（`/cmy/cmy/ranAudio`）。  
> **v1.1**：wav → MiniCPM5-2B（`asr` + `need_fc`；无 FC 时自带 `reply`）→ 仅在需要时调 Qwen Slave。  
> **v1.0 基线保留**：直接出 MiniCPM5 XML FC（`runs/fc/`，22/22）。  
> **不**接入 Demo Gateway `:8006` / `py_backend` 的 1s 全双工循环。

文档：

- [架构说明](docs/architecture.md)（当前实现：v1.1 离线单轮）
- [目标整体链路 §3.2](docs/MiniCPM5-2B_音频门控架构定义.md)（架构定义中，模型槽位未填）
- [版本索引](docs/README.md)
- [v1.0 契约](docs/v1.0.md)（XML FC 基线）
- [v1.1 契约](docs/v1.1.md)（判官）

---

## 仓库关系

```text
/cmy/cmy/MiniCPM-o-Demo   权重、车机 wav、Qwen Slave / ToolRegistry
/cmy/cmy/ranAudio         本仓库：代码、data/、runs/
```

路径常量在 [`ranAudio/paths.py`](ranAudio/paths.py)：

- `DEMO_ROOT`：默认 `../MiniCPM-o-Demo`，可用环境变量 `MINICPMO_DEMO` 覆盖
- `RANAUDIO`：本仓库根（`data/`、`runs/`、`tmp/`）

---

## 定位

| | MiniCPM-o 全双工 FC | ranAudio v1.0 | ranAudio v1.1 |
|--|---------------------|---------------|---------------|
| 交互 | 1s unit + listen/speak | 单轮 → XML FC | 单轮 → 判官 JSON |
| 2B 输出 | — | `<function>…` | `need_fc=true`：asr；`false`：asr+reply |
| 工具规划 | MiniCPM + Slave | 2B 自己填槽 | **仅 need_fc 时** Qwen Slave 规划 |

拍板：无 FC 时 2B **自己当普通 LLM** 回复；有 FC 时 2B **只出 ASR**，交给 Slave。

---

## 环境

- conda：`minicpm5`（自 `qwen3-compare` 克隆，**不要**去升 `qwen3-asr`）
- 解释器：`/cmy/cmy/miniconda3/envs/minicpm5/bin/python`
- Hugging Face：`HF_ENDPOINT=https://hf-mirror.com`，`HF_HUB_DISABLE_XET=1`
- 权重（在 Demo 仓库）：`models/MiniCPM5-2B`；APM 自 `models/MiniCPM-o-4_5`

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
# 若 Demo 不在默认平级路径：
# export MINICPMO_DEMO=/cmy/cmy/MiniCPM-o-Demo
PY=/cmy/cmy/miniconda3/envs/minicpm5/bin/python
cd /cmy/cmy/ranAudio
```

命令在 **本仓库根目录** 执行（`python -m ranAudio.*`）。

---

## v1.1 判官：数据 / 脚本

```bash
# 负例 wav（若尚未生成）
$PY -m ranAudio.gen_chat_tts
# 合并正例(车机)+负例(闲聊，含 reply 模板)
$PY -m ranAudio.build_judge_samples
# -> data/judge_audio.jsonl

# 训练（可选，当前不强制）
# $PY -m ranAudio.train_judge

# 评测（需已有 runs/judge 权重）
# $PY -m ranAudio.eval_judge

# 单条推理：无 FC 直接用 2B reply；有 FC 才打 Slave
$PY -m ranAudio.infer --wav $MINICPMO_DEMO/finetune_fc/demo_car_fc/wavs/navigate_to_000.wav --no-downstream
```

默认 `MINICPMO_DEMO` 未设时，车机 wav 在 `../MiniCPM-o-Demo/finetune_fc/demo_car_fc/wavs/`。

---

## v1.0 基线（保留）

```bash
$PY -m ranAudio.build_samples
$PY -m ranAudio.train_align
$PY -m ranAudio.train_fc
$PY -m ranAudio.eval_car_fc   # 22/22 PASS
```

不要用 v1.1 判官权重去跑 `eval_car_fc`。

---

## FastLM 训练（先测冲突，再 ASR，再路由）

架构仍是 `MiniCPM5Audio`（**SenseVoice-Small** encoder → projector → MiniCPM5-2B）。WenetSpeech 清单来自平级仓库 `../real-asr`，**不**接入 Zipformer fbank。v1.0 FC 对照仍用 Whisper APM。

```bash
# 0) 基线：裸 2B 文本 FC / 语音 FC / 转写×路由×闲聊冲突
$PY -m ranAudio.eval_baseline
# 可加 --max-cases 2 做冒烟

# 1) 导出 WenetSpeech-S 紧凑 jsonl（流式读 Lhotse，不把 151 万行装进内存）
$PY -m ranAudio.prepare_wenet --split wenet-ms --subset S --index
# S1.5：冻结 2B，训 encoder + projector
$PY -m ranAudio.train_s15_asr --prepare-s
$PY -m ranAudio.eval_asr --split aishell-test --also-car

# 2) 路由样本 + S2 LoRA（<asr> 降权）
$PY -m ranAudio.build_route_samples
$PY -m ranAudio.train_s2_route --phase mix --asr-weight 0.2
$PY -m ranAudio.eval_route

# S2 前扫 <asr> 权重（短训）
$PY -m ranAudio.sweep_asr_weight --max-samples 200 --max-steps 200
```

产物：`runs/s15_asr/`、`runs/s2_route/`、`data/wenet_s.jsonl`、`data/route_audio.jsonl`。

---

## 产物路径

| 产物 | 路径 |
|------|------|
| 判官样本 | `data/judge_audio.jsonl` |
| 闲聊 wav | `data/chat_wavs/` |
| Judge LoRA | `runs/judge/`（开训后） |
| v1.0 FC LoRA | `runs/fc/adapter_final/` |
| Align projector | `runs/align/projector.pt` |
| S1.5 encoder/projector | `runs/s15_asr/` |
| S2 route LoRA | `runs/s2_route/` |
| 评测 JSON | `tmp/` |
