# FastLM 偏实时改造（流式运行时）

> **日期**：2026-11-05  
> **性质**：[`STEP_ROUTE.md`](../STEP_ROUTE.md) 的增量实现说明。协议脊柱（§1–§6）不变，只新增流式编排与可插拔 FastLM 槽位。  
> **交付**：可跑原型 + 改造前后时延实测对比。本文随每个里程碑同步更新。

---

## 1. 路线选择

| 路线 | 本轮 | 理由 |
|------|------|------|
| Step 整句底座 + 流式解码 + L2 编排 | **主交付** | 不重训、不换权重；`StepAudio2` 暴露 `.llm` / `.llm_tokenizer` / `.apply_chat_template`，可在 workspace 内自行拼 `generate_inputs` 并挂 streamer，无需改外部目录 |
| MiniCPM5 S2 首 token 门控 + 前缀 chunk 探针 | **对照组 / 硬实时台阶** | `runs/s2_fc_nofc_full/mix_llmcont` 已加齐六个 S2 special token，route_acc 1.0；首 token 是**真 token 级门控** |
| 80–160ms chunk-causal 硬实时 | **后续阶段** | `SenseVoiceEncoderSmall.forward` 是整段前向，`forward_chunk` 从未接入 `SenseVoiceAPM.encode_wavs`；`docs/S1.5_SenseVoice_训练计划.md` L156 明确「不训流式 chunk-causal（S1）」。不重训 S1 直接切 chunk 会破坏已验的 route 1.0，且 S1 是独立训练工程，无法在本轮「原型 + 时延报告」里验证 |

本轮用「前缀 chunk 增量重编码 + 只生成 1–2 token 的路由探针」近似边听边出路由，复杂度 O(k²)，报告中标注精度与复杂度边界。

---

## 2. 基线（改造前，2026-11-02 真实 GPU 实测）

来源：`/tmp/step_route_results.json`（`python -m ranAudio.step_route` 离线同步链路）。

| wav | route | fast_s | slow_s | reinject_s | total_s |
|-----|-------|--------|--------|------------|---------|
| `enable_bluetooth_000` | fc | 3.752 | 2.027 | 1.090 | 6.869 |
| `web_search_stock` | fc | 1.307 | 0.841 | 0.905 | 3.052 |
| `weather_beijing` | fc | 2.086 | 0.911 | 1.407 | 4.404 |

三条全部 `route=fc`，无 NOFC 快闭环样本；三次回注无一输出合规 `<|NOFC|>`，被 `route=other` 降级后仍写进 `user_text`。股价句把「这是演示结果」洗成「1234.56 美元」，回注事实保真失效。

`bench_realtime.py --mode baseline` 汇总（实跑）：`total_s` mean 4.775 / p50 4.404 / min 3.052 / max 6.869；`fast_s` mean 2.382。

**固定基准 wav 集**（`bench_realtime.py::BASELINE_WAVS`，与 `demo_step_route.py::_default_wavs()` 对齐）：

- `/cmy/cmy/MiniCPM-o-Demo/finetune_fc/demo_car_fc/wavs/enable_bluetooth_000.wav`
- `/tmp/step_complex_fc/wavs/web_search_stock.wav`
- `/tmp/step_complex_fc/wavs/weather_beijing.wav`
- `/tmp/step_complex_fc/wavs/datetime_now.wav`

---

## 3. 依赖清单与影响面（里程碑 1 核查）

**现有调用点**（改动签名会受影响）：

| 符号 | 调用点 |
|------|--------|
| `StepFastLM.{__init__,load,route,reinject,_generate}` | `demo_step_route.py:72-73`、`pipeline.py:55,109`；`_generate` 耦合外部 `StepAudio2.__call__` 三元组返回 |
| `StepRoutePipeline.{__init__,run}` / `TurnResult` | `demo_step_route.py:82-87,93-95`、`__init__.py:6` |
| `SlowLMClient.{start,speak,close}` | `demo_step_route.py:77,79,99`、`pipeline.py:92`；stdin/stdout JSON-lines 协议 |
| `parse_s2_output` / `strip_nofc_prefix` / `ROUTE_SYSTEM` / `REINJECT_SYSTEM` | `pipeline.py`、`fastlm.py`、`test_protocol_step.py`、`export_step_sft.py`、`gen_step_backend.py` |
| `demo_step_route.main()` | `__main__.py:3-5`（必须保持无参） |

**结论**：保留同步 `StepRoutePipeline` 作基线不动，新增 `RealtimePipeline`；所有新行为由 `--mode` 显式开启；`TurnResult.to_dict()` 原键保持兼容（新字段追加在末尾）。

**权重/数据**（均已存在）：`runs/s2_fc_nofc_full/mix_llmcont/llm_final.safetensors`、`runs/s15_asr/b_ms_llm_cont/{encoder_best,projector_best}.pt`、`/cmy/cmy/MiniCPM-o-Demo/models/SenseVoiceSmall/model.pt`、`/cmy/cmy/Step-Audio/Step-Audio-2-mini`、`/cmy/cmy/Qwen/Qwen3.5-2B`。

**环境硬约束（2026-11-05 实测）**：

| 项 | 现状 | 影响 |
|----|------|------|
| 解释器分裂 | Step venv = transformers 4.49.0 / hub 2.4.0；`minicpm5`、`qwen3-compare` = transformers 5.15.0.dev0 / hub 1.27.0 | **两侧必须分进程**；MiniCPM5 后端走独立 worker 子进程 |
| Step venv 依赖冲突 | `import transformers` 报 `huggingface-hub>=0.26.0,<1.0 ... found 2.4.0` | Step 基座本轮无法加载；`step-stream` 基准产出结构化 error 行，不伪造数据 |
| GPU | H800 80GB，已占 ~70.6GB，可用 ~10.4GB | Step-Audio-2-mini（~16.5GB bf16）放不下；MiniCPM5-2B 实跑通过（曾因 GPU 暂存 state_dict OOM，改为 CPU 加载后解决） |
| Step 模型实现 | `modeling_step_audio_2.py:361-365` 返回 `past_key_values=None`；`forward` 每步重插音频特征 | streamer 可行但每 token 全量重算 prefill，收益来自提前 emit，**不缩短编码** |
| `padding_mels` | 来自 `/cmy/cmy/Step-Audio/Step-Audio2/utils.py:71-91`（仓库外） | workspace 内实现等价 padding，不改外部目录 |
| `_boot.py` | 纯 sys.path，不引入 torch/CUDA | 纯解析逻辑做零 torch 回退；`test_realtime.py` 可无 GPU 运行 |

**测试风格**：`test_protocol_step.py` 是 `main() + 裸 assert`，无 pytest；`test_realtime.py` 沿用，10 组断言。

---

## 4. 运行命令

```bash
cd /cmy/cmy/ranAudio
PY=/cmy/cmy/Step-Audio/venv/bin/python

# 四模式基准（同集 / 各自产出 JSON + Markdown 报告）
env -u PYTHONPATH $PY -m ranAudio.step_route.bench_realtime --mode baseline \
  --out /tmp/realtime_baseline.json --report /tmp/realtime_baseline.md
env -u PYTHONPATH $PY -m ranAudio.step_route.bench_realtime --mode offline-events \
  --out /tmp/realtime_offline.json --report /tmp/realtime_offline.md
env -u PYTHONPATH $PY -m ranAudio.step_route.bench_realtime --mode step-stream \
  --wav /tmp/step_complex_fc/wavs/weather_beijing.wav --out /tmp/realtime_stepstream.json
env -u PYTHONPATH PYTORCH_ALLOC_CONF=expandable_segments:True $PY \
  -m ranAudio.step_route.bench_realtime --mode minicpm-probe \
  --wav /tmp/step_complex_fc/wavs/weather_beijing.wav \
  --wav /tmp/step_complex_fc/wavs/datetime_now.wav \
  --out /tmp/realtime_minicpm.json --report /tmp/realtime_minicpm.md

# demo 的流式分支（默认 baseline 行为不变）
env -u PYTHONPATH $PY -m ranAudio.step_route.demo_step_route \
  --mode offline-events --out /tmp/demo_offline.json --report /tmp/demo_offline.md

# no-GPU 单测
env -u PYTHONPATH $PY ranAudio/step_route/test_realtime.py
env -u PYTHONPATH $PY ranAudio/step_route/test_protocol_step.py

# MiniCPM5 首 token 门控 / 前缀 chunk 探针（minicpm5 环境直跑）
env -u PYTHONPATH PYTHONPATH=/cmy/cmy/ranAudio PYTORCH_ALLOC_CONF=expandable_segments:True \
  /cmy/cmy/miniconda3/envs/minicpm5/bin/python -m ranAudio.step_route.fastlm_minicpm \
  --wav /tmp/step_complex_fc/wavs/weather_beijing.wav --out /tmp/minicpm_route_bench.json
# 加 --probe 切到前缀 chunk 探针
```

四模式均已实跑，产物见 §7。

---

## 5. 流式核心契约

| 模块 | 作用 |
|------|------|
| `events.py` | `StreamEvent`（kind / request_id / t_ms / text / route / meta）、`Utterance`（float32 mono 16k PCM + `save_wav`，文件名保留原 stem）、`Clock`、`segment_latency` |
| `stream_protocol.py` | `IncrementalParser`：逐 token 累积，容忍半截 special token，首路由 tag 完整即发 `route_first`，正文/转写发 delta，`finish()` 发 `route_final` + `done`；`guard_speak()` 空白归一化后校验 speak 逐字保真，不通过则结构化拼接兜底 |
| `vad.py` | `VADConfig`（尾静音 500ms / 最短 300ms / max_utt 15s / 25ms 帧）、`energy_vad`、`split_utterances`、`VADStream`（增量 feed + `barge_in` + `flush`）、`load_wav_mono16k` |
| `session_store.py` | §5 六字段 + 多活跃请求；`independent_turns=True` 时批量基准不误取消在途慢结果，`False` 时新请求 cancel 旧请求（barge_in）；`arbitrate` 返回 speak/hold/drop |
| `fastlm_base.py` | `FastLMBackend` Protocol、`stream_text`（支持 prefill 时间偏移）、`RecordedBackend`（回放历史输出并注入实测 `fast_s/slow_s/reinject_s`，使无 GPU 基准可解释） |
| `fastlm_stream.py` | `StepStreamBackend`：workspace 内拼 `generate_inputs` + `TextIteratorStreamer` 线程化 generate；streamer 不可用时整段 generate 回放 delta |
| `fastlm_minicpm.py` | `--serve` worker（minicpm5 环境持模型）+ `MiniCPMProbeBackend`（协调进程，JSON-lines）；首 token 真门控、前缀 chunk 探针、结构化回注 |
| `coordinator.py` | `RealtimeRunner`：FC+BASE 读到即 emit → 线程池异步慢路径 → Store → 仲裁 → 回注事件 → `guard_speak` → 单一播报出口；`hold`/`drop`、request_id/cancel、history |
| `pipeline.py` | 同步 `StepRoutePipeline` 一字未改作基线；新增 `RealtimePipeline` 流式入口；`TurnResult` 追加分段时延字段 |
| `slowlm.py` | `speak(..., request_id=)` 透传 id、worker 回包带同一 id；`cancel(rid)`；迟到/取消结果记入 `_late` 而非静默丢弃；worker 识别 `cmd=cancel` |
| `protocol_step.py` | `_cut_body` 不再把 user 侧 `<backend>` 当停止符（此前会截断合法正文）；新增 `verify_backend_fact` 入口 |
| `demo_step_route.py` | `--mode` / `--report` 分支，默认 baseline 行为不变 |
| `bench_realtime.py` | 四模式 harness、分段时延汇总、保真兜底统计、基线对比表；模型环境不可用时输出结构化 error 行 |
| `test_realtime.py` | 10 组 no-GPU 单测（含多活跃请求、回放、事实保真空白归一化） |

### 5.1 Step 后端的环境回退

`StepStreamBackend` 不修改 `/cmy/cmy/Step-Audio` 任何文件：只调用 `apply_chat_template` 拿 `(results, mels)`，在 workspace 内做 tokenize、`pad_sequence`、`transpose`，再调 `.llm.generate(..., streamer=...)`。本轮 `step-stream` 基准因 hub 冲突实跑失败（`ImportError: huggingface-hub>=0.26.0,<1.0 ... found 2.4.0`），报告记录为 error 行；把 Step venv 的 hub 降回 0.3x（本机其它环境有 0.36.2）并等 GPU 空闲后，同一命令即可复跑。

### 5.2 MiniCPM5 首 token 门控（真实 GPU 实测）

`fastlm_minicpm.py` 分两侧：`--serve` worker 跑在 `minicpm5` 环境持有模型，`MiniCPMProbeBackend` 留在协调进程通过 JSON-lines 通信。权重加载曾因在 GPU 上暂存 4.7GB state_dict 触发 OOM，改为 CPU 加载再逐参数复制后跑通。

**整句端点后首 token**（`/tmp/minicpm_route_bench.json`，4 个基准 wav，同一次加载）：

| wav | 首 token | first_token_ms | full_text | total_ms | warmup |
|-----|----------|----------------|-----------|----------|--------|
| `enable_bluetooth_000` | `<\|FC\|>` | 681.9 | `<\|FC\|><\|BASE\|>打开蓝牙` | 2649.1 | 是 |
| `web_search_stock` | `<\|FC\|>` | 115.8 | `<\|FC\|><\|BASE\|>帮我搜一下特斯拉最新股价` | 417.0 | 否 |
| `weather_beijing` | `<\|FC\|>` | 91.4 | `<\|FC\|><\|BASE\|>查一下北京明天的天气` | 350.9 | 否 |
| `datetime_now` | `<\|FC\|>` | 83.1 | `<\|FC\|><\|BASE\|>现在几点了` | 272.3 | 否 |

稳态首 token **83–116ms**，落在 80–160ms 目标区间；4/4 路由正确、BASE 后转写正确（对比 Step 基线的助手口吻与编造天气）。注意这是**整句 wav 已完整输入后的 prefill**，不是边听边出。

**前缀 chunk 探针**（`/tmp/minicpm_probe_chunks.json`，`weather_beijing`，250ms/chunk，早停于 chunk 4）：

| chunk | audio_ms | route | t_ms | since_start_ms |
|-------|----------|-------|------|----------------|
| 1 | 250 | fc | 1029.5（warmup） | 2352.8 |
| 2 | 500 | nofc | 105.4 | 2458.3 |
| 3 | 750 | fc | 99.1 | 2557.4 |
| 4 | 1000 | fc | 122.2 | 2679.7 |

稳态每 chunk **99–122ms**，但首个 chunk 有 1.03s CUDA warmup；路由在 500ms 前缀抖动一次，750ms 后稳定；累计 2.68s（含 warmup）。该探针是 O(k²) 前缀重编码近似，不是 chunk-causal：真 80–160ms 边听边出仍需 S1 重训 + AEC。

**回注**：MiniCPM5 S2 未训 backend 拼接桶，worker 直接返回结构化拼接 `f"{NOFC}{speak}"`（`structured_fallback=True`），不让生成模型裸拷贝事实。

### 5.3 L2 异步编排语义

- FC+BASE 读到即 emit，主循环立刻回到听；慢任务在线程池执行，`SlowLMClient.speak(transcript, request_id=rid)` 带回同一 id。
- `SessionStore.set_pending` 拒收过期/取消 id；`arbitrate(rid, user_speaking)` 按「用户优先」给 speak/hold/drop。批量基准用 `independent_turns=True` 避免相邻句子互相误取消；真实 barge_in 用 `False`。
- 回注稿先过 `guard_speak`：speak 未逐字出现（空白归一化后）就退回「衔接词 + 原样 speak」，确认类事实不加「查到了」。
- `user_text` 是唯一播报出口；SlowLM 的 speak 从不直写该字段。
- 无 AEC / 真实麦克风 / 流式 TTS：VAD 是能量端点替身，播报 sink 只记文本。

---

## 6. 里程碑状态

| 里程碑 | 状态 | 交付 |
|--------|------|------|
| 1 基线固化与影响面 | **完成** | `bench_realtime.py`（baseline 实跑）、本文、`STEP_ROUTE.md` §7/§9 同步 |
| 2 流式核心 | **完成** | `events.py` / `stream_protocol.py` / `vad.py` / `session_store.py` / `test_realtime.py` |
| 3 可插拔协议 + Step 流式后端 | **完成（代码）** | `fastlm_base.py` / `fastlm_stream.py` / `fastlm.py` 增量改造；模型级实跑受 §3 环境约束阻塞 |
| 4 MiniCPM5 首 token 门控 + chunk 探针 | **完成（已实跑）** | 4 个 wav 首 token 83–116ms、4/4 路由正确；前缀探针每 chunk 99–122ms |
| 5 L2 异步编排 + request_id / cancel | **完成（已实跑）** | `coordinator.py` + 5 个文件增量改造；`offline-events` / `minicpm-probe` 全链路跑通，10 组单测通过 |
| 6 三模式基准与对比报告 | **完成** | 四模式 JSON + Markdown 报告；下文 §7 为汇总结论 |

---

## 7. 改造前后对比报告（实测汇总）

产物：`/tmp/realtime_baseline.{json,md}`、`/tmp/realtime_offline.{json,md}`、`/tmp/realtime_minicpm.{json,md}`、`/tmp/realtime_stepstream.{json,md}`。

### 7.1 同集三条 FC：同步基线 vs 流式编排（回放注入真实模型耗时）

| wav | baseline `total_s` | offline-events `total_s` | Δ | `fast_loop_s`（到 emit） | 回注保真 |
|-----|--------------------|--------------------------|---|--------------------------|----------|
| `enable_bluetooth_000` | 6.869 | 6.878 | +0.009 | 3.757 | L2 兜底修正 |
| `web_search_stock` | 3.052 | 3.058 | +0.006 | 1.309 | L2 兜底修正 |
| `weather_beijing` | 4.404 | 4.411 | +0.007 | 2.089 | 逐字保真 |

**读法**：保留 Step 底座、不重训时，端到端时延**几乎不变**（+6–9ms 是编排开销），因为 prefill 与两次模型前向都还在。收益有两处：

1. 慢路径不再占快时钟——`fast_loop_s` 只到 emit（1.3–3.8s），SlowLM 与回注在后台，听不阻塞；`total_s` 是最终播报时刻，二者分离。
2. 回注事实保真被修复——基线 3/3 回注不合规（股价句把「这是演示结果」洗成「1234.56 美元」），新链路 2 次触发结构化兜底、1 次逐字保真，`user_text` 不再含生成模型编造的数字。

`offline-events` 汇总（4 条，含 1 条无历史回放的 NOFC）：`total_s` mean 3.587 / p50 3.734；`fast_loop_s` mean 1.789；`route_first_ms` mean 1786（≈ fast_s，说明 Step 整句底座的首路由仍然晚）。

### 7.2 换首 token 门控底座：真提速

`minicpm-probe` 全链路（`MiniCPMProbeBackend` 跨解释器 worker，无 SlowLM 挂载、结构化回注）：

| wav | `route_first_ms` | `fast_s` | `total_s` | 备注 |
|-----|------------------|----------|-----------|------|
| `weather_beijing` | 1307.0 | 3.199 | 3.203 | CUDA warmup |
| `datetime_now` | **83.8** | **0.280** | **0.281** | 稳态 |

稳态首路由从 Step 的 1.3–3.8s 降到 **83.8ms**，端到端 0.28s（约一个数量级）。需诚实标注：这里换了 FastLM 底座（MiniCPM5 S2 权重），且慢路径是占位文本、回注是结构化拼接，并非同一负载下的纯编排收益。

### 7.3 环境阻塞与边界

| 模式 | 结果 |
|------|------|
| `step-stream` | 环境阻塞，1 条结构化 error：`ImportError: huggingface-hub>=0.26.0,<1.0 ... found 2.4.0`；GPU 余量也不足以加载 Step。代码与整段 generate 回退路径就绪，未伪造时延 |
| 前缀 chunk 探针 | 稳态 99–122ms/chunk，但首 chunk 1.03s warmup、累计 2.68s；O(k²) 重编码，不是 chunk-causal |
| AEC / 真麦克风 / 流式 TTS | 未做。常开麦下无 AEC 会把自身播报回采成新命令，是真全双工的前置条件 |

### 7.4 结论

**「把 FastLM 改成偏实时」在换到首 token 门控底座时成立**（稳态 83.8ms 首路由 / 0.28s 端到端）；**保留 Step 整句底座时只能做到「不阻塞」而非「更快」**——首路由仍受整句 prefill 限制，端到端时延基本不变，但快时钟解脱、回注事实保真被 L2 修好。真 80–160ms 边听边出仍需 S1 chunk-causal 重训 + AEC，不在本轮范围。

---

## 8. 后续阶段（本轮不做）

1. S1 chunk-causal 重训 + AEC：真 80–160ms 硬实时的前置条件。
2. Step SFT + 词表加齐六个 special token：让 `<|FC|>` / `<|NOFC|>` 成为真 token 门控，摆脱字符串软恢复；同时修复 Step venv 的 hub 冲突并复跑 `step-stream` 基准。
3. 挂载真实 SlowLM 与工具后端，把 `minicpm-probe` 的占位 speak 换成真实结果，重测同负载时延。
4. 真实麦克风 / VAD 模型 / 流式 TTS 与回采参考。
5. 回注事实保真长期靠 SFT 收敛，L2 校验保留为安全网。
