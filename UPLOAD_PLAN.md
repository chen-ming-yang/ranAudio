# 上传决策总表 · `/cmy/cmy` 资产分类与处置方案

> 生成：2026-11-21　复核：本文件所有体积均为 `du`/`stat` 实测，非估算
> 目的：把 3.9 TB 资产拆成「**可公开**／**需私有备份**／**可直接丢弃**」三类，
> 由你逐项勾选，再按勾选结果执行上传。

---

## 0. 先回答你问的两个问题

### 0.1 代码都提交到 GitHub 了吗？—— **没有，还差 12 个 commit**

六仓库工作区已全部干净（无未提交改动），但每个仓库都有 **2 个本地 commit 尚未推送**：

| 仓库 | 待推送 commit | 远程 |
|------|---------------|------|
| `MeanSE` | `4e32a56` 补 LICENSE/NOTICE<br>`7f1abbd` 更新数据/推理/训练脚本 | `github.com/chen-ming-yang/MeanSE` |
| `ranAudio` | `226b6a7` 补 LICENSE/第三方声明<br>`b97cc64` WenetSpeech-L 三方一致数据 + kimi 文字规范 | `git@github.com:chen-ming-yang/ranAudio` |
| `ranAudio/cdsd` | `7ab6e72` 补 LICENSE/NOTICE<br>`0c5c086` MoE 双视图个性化 ASR | `github.com/chen-ming-yang/cdsd` |
| `speechsec` | `3f3c3ff` 补 MIT LICENSE<br>`938dd16` 本地实验笔记 | `git@github.com:chen-ming-yang/speech-sec` |
| `real-asr` | `775636d` 补 LICENSE/NOTICE<br>`b49fcad` RNNT/CTC 配置 + icefall 编码器 | `github.com/chen-ming-yang/real-asr` |
| `Qwen3-ASR` | `bf9b5b2` 新增 NOTICE<br>`5e08156` LoRA 微调脚本 + 0614 数据清单 | `git@github.com:chen-ming-yang/Qwen3-ASR` |

> 每个仓库的前一条 commit 是**更早的代码改动**，一直没推送；后一条才是今天的许可证。
> 也就是说：**不只许可证没上 GitHub，之前的代码改动也还没上。**

其余仓库状态（供参考，多数是第三方 fork 或无需处理）：

| 仓库 | 状态 |
|------|------|
| `MeanFlowSE` / `MeanFlowSE123` | 干净，`MeanFlowSE123` 有 2 条未推 |
| `Qwen` | 干净，有 2 条未推 |
| `FunASR`（你的 fork） | 干净，有 5 条未推 |
| `DARS` | 21 个改动，**无 remote**（checkpoints 已确认丢弃） |
| `MiniCPM-o-Demo`（OpenBMB 上游） | 29 个改动，不建议推 |
| `rmsaqa-code` / `SemScore` / `UUVC` / `WenetSpeech` / `compute-wer` / `Qwen3-TTS` / `FireRedASR2S` | 均为他人仓库，本地改动无需推送 |

### 0.2 当前已完成的发布

| 内容 | 渠道 | 状态 |
|------|------|------|
| Tier-A 三方一致数据集（10 分片 / 1,015.6 万条 / 6,190.72 h / 3.66 GiB） | ModelScope `chenhaha12315/wenet-speech-L-filter` | ✅ 已发布 |
| 8 个模型发布包（95.47 GiB / 357 文件） | 本地 `/cmy/publish_ms` 暂存 | ⏸ 待你确认后上传 |

---

## 1. 三个渠道的能力边界

| 能力 | GitHub | ModelScope | 阿里云 OSS `oss://ranyan` |
|------|--------|-----------|---------------------------|
| 单文件上限 | 100 MB（>50 MB 警告） | **50 GB** | 无（分片上传） |
| Git LFS 单文件 | 2 GB | — | — |
| 仓库/存储总量 | ~1 GB 建议 | 非 LFS 文件 ≤ 500 MB，文件数 ≤ 10 万 | 按量付费 |
| 是否公开 | 公开 | 公开（可设私有） | **私有** |
| 适合放 | 代码、脚本、小配置 | 模型权重、公开数据集 | 全量备份、原始音频 |
| 实测速率 | — | 受 Hub 限制 | **11.19 MiB/s（约 90 Mbps）** |

**结论**：GitHub 只放代码；模型权重走 ModelScope；剩下的全部走 OSS 私有备份。

---

## 2. A 类 · 公开模型权重 → ModelScope

> 已在 `/cmy/publish_ms` 构建完毕，**全部为硬链接，零额外磁盘占用**。
> 已剔除 `optimizer.pt`、`trainer_state.json`、`training_args.bin`、`rng_state.pth`、
> `scheduler.pt`、`*.gguf`、`stale_base_shards/`。
> 每个仓库均已包含 `LICENSE`（适用时含 `NOTICE` / `THIRD_PARTY_NOTICES.md`）。

### A1. `qwen3-asr-ft-wenetspeech` — Qwen3-ASR 微调合集

| 项 | 内容 |
|---|---|
| **是什么** | 基于 Qwen3-ASR-1.7B 在 WenetSpeech-L / AISHELL / 构音障碍数据上的 9 组微调产物 |
| **体积** | **34.89 GiB** / 109 文件 / 9 个权重 |
| **结构** | `finetune-again-20260512/`、`finetuning-all-out-20260524/`、`finetuning-all-out-20260616/`、`finetuning-best/checkpoint-10432/`、`finetuning-out-20260509/`、`finetuning-out-20260524/`、`finetuning-out-20260615/`、`finetuning-out-bak/`、`gspo-final/` |
| **单份** | 3.80 GB × 8，`gspo-final` 4.39 GB（均只含 `model.safetensors` + config + tokenizer） |
| **用途** | 中文 ASR 微调基线、Tier-A 数据生成所用的模型 |
| **许可证** | Apache-2.0（`LICENSE` 保留 Qwen 上游原文，`NOTICE` 划清本 fork 改动范围） |
| **建议** | ✅ **P0 上传**。这是最有价值的模型资产 |
| **勾选** | ☐ 上传　☐ 跳过 |

### A2. `ranAudio-s2-full` — S2 全参微调（7 个分支）

| 项 | 内容 |
|---|---|
| **是什么** | MiniCPM5-2B 音频门控架构 S2 阶段的全参数微调权重，7 个实验分支各一份最终模型 |
| **体积** | **32.95 GiB** / 37 文件 / 14 个权重 |
| **结构** | `asr/`、`chat/`、`route/`、`mix/`、`mix_boundary/`、`mix_llmcont/`、`mix_proj/`（每个含 `llm_final.safetensors` 4.69 GB + `projector_final.pt` + `tokenizer/`） |
| **用途** | 语音门控：判断「这句话该走 ASR 还是直接回复」；`mix_boundary` 为当前最佳 |
| **许可证** | Apache-2.0 + `THIRD_PARTY_NOTICES.md` |
| **建议** | ⚠️ **P2**。体积大、分支多；若只想公开最佳，可只传 `mix_boundary/`（约 4.7 GiB） |
| **勾选** | ☐ 全部上传（32.95 GiB）　☐ 只传 `mix_boundary`　☐ 跳过 |

### A3. `cdsd-moe-lora` — MoE 双视图个性化 ASR

| 项 | 内容 |
|---|---|
| **是什么** | 在 Qwen3-ASR 上做「稠密 LoRA + MoE 专家」的个性化 ASR 实验产物 |
| **体积** | **7.81 GiB** / 106 文件 / 16 个权重 |
| **结构** | `moe_spk01/04/06/08/09/12/all/`，每个含 `dense_lora/adapter_model.safetensors`（~20 MB）+ `moe8_lam*/moe_state.pt`（~750 MB）；`moe_spk06` 另有 `moe16_lam05`（850 MB）与 4 组 λ 消融 |
| **效果** | 相对 dense 基线 CER **约 −6 pp**，最优 `moe8_lam1` |
| **用途** | 说话人个性化 ASR、MoE 参数高效微调复现 |
| **许可证** | Apache-2.0 + `NOTICE` |
| **建议** | ✅ **P0 上传**。体积适中、结论明确 |
| **勾选** | ☐ 上传　☐ 只传 `moe_spk06`（主实验 + λ 消融，约 4.6 GiB）　☐ 跳过 |

### A4. `ranAudio-s2-lora` — S2 阶段 LoRA（含 projector）

| 项 | 内容 |
|---|---|
| **是什么** | 与 A2 同架构，但用 LoRA 代替全参微调；3 个任务分支 × 2 个 checkpoint |
| **体积** | **6.10 GiB** / 23 文件 / 9 个权重 |
| **结构** | `asr/`、`chat/`、`route/`，各含 `adapter_epoch1/`、`adapter_final/`（1.01 GB）+ `projector_final.pt` |
| **用途** | 低成本复现门控能力；显存占用远低于 A2 |
| **许可证** | Apache-2.0 + `THIRD_PARTY_NOTICES.md` |
| **建议** | ✅ **P0 上传**。和 A2 互为对照，实用价值高 |
| **勾选** | ☐ 上传　☐ 跳过 |

### A5. `ranAudio-s15-asr` — S1.5 ASR 变体（8 组）

| 项 | 内容 |
|---|---|
| **是什么** | SenseVoice encoder + projector 的 8 组 ASR 变体（不同数据配比 / 是否接 LLM / 构音障碍增强） |
| **体积** | **5.02 GiB** / 32 文件 / 14 个权重 |
| **结构** | `a/`、`b_s/`、`b_ms/`、`b_ms_cont/`、`b_ms_llm/`、`dysarthric/`、`dysarthric_e2/`、`smoke/`（各含 `encoder_best.pt` 0.82 GB + `projector_best.pt`） |
| **用途** | 流式 ASR 前端；`dysarthric*` 针对构音障碍 |
| **许可证** | Apache-2.0 + `THIRD_PARTY_NOTICES.md` |
| **建议** | ✅ **P1 上传** |
| **勾选** | ☐ 上传　☐ 跳过 |

### A6. `MeanSE-meanflow` — MeanSE 语音增强

| 项 | 内容 |
|---|---|
| **是什么** | ICASSP 2026 投稿工作 *MeanSE: Efficient Generative Speech Enhancement with Mean Flows* 的训练产物 |
| **体积** | **4.93 GiB** / 20 文件 / 4 个权重 |
| **结构** | `ncsnpp/version_0/checkpoints/{best_epoch=05-...,last}.ckpt`、`ncsnpp_dysar/version_0/checkpoints/{best_epoch=21-...,last}.ckpt`（各 1.23 GB） |
| **用途** | 1-NFE 生成式语音增强；`ncsnpp_dysar` 为构音障碍域 |
| **论文** | https://arxiv.org/abs/2509.21214 |
| **许可证** | Apache-2.0 + `NOTICE`（Google Research `ncsnpp.py`、ESPnet、Urgent2026） |
| **建议** | ✅ **P1 上传**。有论文支撑，曝光价值高 |
| **勾选** | ☐ 上传　☐ 只传 `best_*.ckpt`（去 `last.ckpt`，约 2.5 GiB）　☐ 跳过 |

### A7. `qwen3.5-0.8b-merged` — 合并后小模型

| 项 | 内容 |
|---|---|
| **是什么** | Qwen3.5-0.8B 经微调 + merge 后的完整模型（两个版本） |
| **体积** | **2.86 GiB** / 27 文件 / 2 个权重 |
| **结构** | `merged/`（1.40 GB）、`merged-v4/`（1.40 GB），各含自带 `LICENSE` |
| **用途** | 端侧轻量基座；v4 为最终版 |
| **许可证** | Apache-2.0（子目录自带） |
| **建议** | ⚠️ **P2**。官方 `Qwen/Qwen3.5-0.8B` 已有高下载量，派生版价值中等 |
| **勾选** | ☐ 上传两个　☐ 只传 `merged-v4`　☐ 跳过 |

### A8. `speechsec-st` — 语音安全模型

| 项 | 内容 |
|---|---|
| **是什么** | SpeechSEC（Interspeech 2025 非官方实现）的单任务最佳 checkpoint |
| **体积** | **0.90 GiB** / 3 文件 |
| **结构** | `best.pt` + `train.log` + `LICENSE` |
| **用途** | 语音合成 / 编辑 / 续写统一框架 |
| **许可证** | **MIT**（本项目独立实现，非官方） |
| **建议** | ⚠️ **P1**。需在模型卡片写明「非官方实现」 |
| **勾选** | ☐ 上传　☐ 跳过 |

### A 类小结

| 勾选组合 | 总体积 |
|---|---|
| 全部 8 个 | **95.47 GiB** |
| 精简版（A1+A3主实验+A4+A5+A6best） | 约 **54 GiB** |
| 最小可用（A1+A4） | 约 **41 GiB** |

---

## 3. B 类 · 公开数据集 → ModelScope Datasets

### B1. WenetSpeech-L Tier-A 三方一致子集 — ✅ 已发布

| 项 | 内容 |
|---|---|
| 仓库 | `chenhaha12315/wenet-speech-L-filter` |
| 规模 | 10 分片 / **10,156,566 条** / **6,190.72 小时** / 3.66 GiB |
| 筛选条件 | 官方标签 = Qwen3-ASR-1.7B = FireRedASR2-AED（三模型完全一致） |
| 许可证 | CC-BY-4.0 + `CITATION.md` |
| 状态 | ✅ 已上线 |

### B2. WenetSpeech-L 多 ASR 伪标签全集（候选）

| 项 | 内容 |
|---|---|
| **是什么** | 对 WenetSpeech-L 全量（L10 + L90）跑 Qwen3-ASR 与 FireRedASR2 的**逐条识别结果** |
| **体积** | `qwen3_asr_L10pct` 515 M + `qwen3_asr_L90pct` 6.1 G + `firered_asr_L10pct` 505 M + `firered_asr_L90pct` 4.5 G = **约 11.6 G**（747 个分片文件） |
| **用途** | 伪标签蒸馏、三模型共识研究、ASR 输出差异分析 |
| **价值** | 中等——重跑需 **约 311 GPU 小时**（Qwen 139 h + FireRed 172 h），故有保留价值 |
| **许可证** | 需沿用 WenetSpeech 的 CC-BY-4.0，且标注「机器伪标签，非人工标注」 |
| **建议** | ⚠️ **可选**。若不做伪标签研究，建议只进 OSS 备份 |
| **勾选** | ☐ 发布到 ModelScope　☐ 只进 OSS　☐ 直接丢弃 |

### B3. 门控 / 对齐 SFT 数据（候选）

| 项 | 内容 |
|---|---|
| **是什么** | `ranAudio/data/` 下的 SFT 清单与切片音频（`s2_tts_wavs` 571 M、`wenet_ms.jsonl` 359 M、`chat_wavs` 59 M、`wenet_s.jsonl` 37 M、`step_s2_mix.jsonl` 21 M、`dysarthric.jsonl` 14 M 等） |
| **体积** | 约 **1.3 G** |
| **用途** | 音频门控 S2 训练、构音障碍 ASR |
| **注意** | 含 LibriTTS-R / WenetSpeech 切片，**再分发需遵守各自协议**；`data/char_align`（159 M / 1270 wav）已在你确认下加入 `.gitignore` |
| **建议** | ❌ **不建议公开**。衍生素材版权链复杂，进 OSS 即可 |
| **勾选** | ☐ 发布（需先做版权梳理）　☐ 只进 OSS |

---

## 4. C 类 · 私有全量备份 → OSS `oss://ranyan`

> 不上 ModelScope、不上 GitHub，只做灾备。实测 11.19 MiB/s（约 90 Mbps）。

| 批次 | 内容 | 体积 | 说明 |
|------|------|------|------|
| C1 | Tier-A 全套（10 分片 + 原始两份 + summary + 中间产物） | ~7.4 G | 重跑需 4 天 GPU + 2.5 h CPU |
| C2 | 全部训练 run 目录（`ranAudio/runs`、`Qwen3-ASR` 各输出、`real-asr/exp`、`rmsaqa-code/models`） | ~400 G | **含 `optimizer.pt` 与中间 ckpt**，这是 ModelScope 包里的删除部分 |
| C3 | 自建数据（manifest、SFT、切片音频） | ~100 G | 可重建但耗时 |
| C4 | `rmsaqa-code/data` | 342 G | 注意 `RMS-AQA` 与 `RMS-AQA-v1-20261104` **疑似重复，需先比对** |
| C5 | 早期实验（`runs/{fc,s2_fc_nofc,align}`、`voice_clone`） | 26 G | 低价值但体积不大 |

**C 类合计约 875 GB**，按 11.19 MiB/s 计时约 **22 小时**。

---

## 5. D 类 · 不上传（重下更快 / 无价值）

| 内容 | 体积 | 理由 |
|------|------|------|
| `WenetSpeech/` 原始音频 | **2.1 T** | 官方公开，新机重下 |
| `speechsec/data` + `LibriTTS_R` | 95 G | 公开数据集，可重下/重算 |
| `FunASR/examples` | 53 G | 公开示例 |
| `DARS/checkpoints` | 48 G | **已确认丢弃** |
| `rmsaqa-code/runs` | 6.1 G | **已确认丢弃** |
| `tmp_audio_equal_len.tar.gz` | 2.6 G | 临时打包 |
| `ossutil` / `mihomo` / Miniconda | ~260 M | 重装即可 |

**D 类合计约 2.3 TB 不需要迁移。**

---

## 6. 勾选汇总表（请在此打勾）

### 6.1 立即要做的

| # | 事项 | 动作 | 勾选 |
|---|------|------|------|
| 1 | 6 个仓库共 12 个 commit 推送到 GitHub | `git push` × 6 | ☐ |
| 2 | 8 个 ModelScope 仓库的 **README 模型卡片** | 我生成，你审 | ☐ |

### 6.2 模型上传（A 类）

| # | 仓库 | 体积 | 勾选 |
|---|------|------|------|
| A1 | `qwen3-asr-ft-wenetspeech` | 34.89 GiB | ☐ |
| A2 | `ranAudio-s2-full` | 32.95 GiB | ☐ |
| A3 | `cdsd-moe-lora` | 7.81 GiB | ☐ |
| A4 | `ranAudio-s2-lora` | 6.10 GiB | ☐ |
| A5 | `ranAudio-s15-asr` | 5.02 GiB | ☐ |
| A6 | `MeanSE-meanflow` | 4.93 GiB | ☐ |
| A7 | `qwen3.5-0.8b-merged` | 2.86 GiB | ☐ |
| A8 | `speechsec-st` | 0.90 GiB | ☐ |

### 6.3 数据集发布（B 类）

| # | 内容 | 体积 | 勾选 |
|---|------|------|------|
| B1 | Tier-A 三方一致 | 3.66 GiB | ✅ 已发布 |
| B2 | 多 ASR 伪标签全集 | 11.6 G | ☐ |
| B3 | 门控 SFT 数据 | 1.3 G | ☐ |

### 6.4 OSS 私有备份（C 类）

| # | 内容 | 体积 | 勾选 |
|---|------|------|------|
| C1 | Tier-A 全套 | 7.4 G | ☐ |
| C2 | 全部训练 run | ~400 G | ☐ |
| C3 | 自建数据 | ~100 G | ☐ |
| C4 | `rmsaqa-code/data`（先去重） | 342 G | ☐ |
| C5 | 早期实验 | 26 G | ☐ |

---

## 7. 建议执行顺序

```
第 1 步  推送 6 个仓库的 12 个 commit          （秒级）
第 2 步  生成并审阅 A 类模型卡片              （可并行）
第 3 步  上传 A1 / A3 / A4（价值最高、体积适中）  （约 49 GiB）
第 4 步  上传 A5 / A6 / A8
第 5 步  视反馈决定 A2 / A7
第 6 步  OSS 备份 C1 → C3 → C2 → C4
```

---

## 8. 已知风险与注意事项

| 风险 | 说明 |
|------|------|
| **ModelScope 需先建仓** | 8 个仓库需在 `chenhaha12315` 下先创建，否则 `upload_folder` 报 404 |
| **单文件 ≤ 50 GB** | 现有最大单文件 4.69 GB，✅ 无风险 |
| **文件数 ≤ 10 万** | 最多 109 文件，✅ 无风险 |
| **`qwen3.5-0.8b-merged` 顶层无 LICENSE** | 仅在 `merged/`、`merged-v4/` 子目录内，建议上传时补一份到顶层 |
| **`speechsec` 非官方实现** | 模型卡片必须写明，避免误导 |
| **GLM-4-Voice 权重** | 自定义协议。**已在 `.gitignore` 排除**，不入库、不上传；代码部分是 Apache-2.0（Zhipu AI），可正常分发 |
| **`data/char_align`** | 159 MB / 1270 wav，已加入 `.gitignore`，不随仓库分发 |
| **`MODELSCOPE_MAP.md` 含绝对路径** | 若公开 `ranAudio` 仓库，需先把 `/cmy/cmy/...` 改为相对路径 |
