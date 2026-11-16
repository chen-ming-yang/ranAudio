# WenetSpeech-L 三方一致（Tier-A Agree3）结果汇总

本文档记录 **官方标签 = Qwen3-ASR-1.7B = FireRedASR2-AED**（去标点/空白后比较）的产物位置、规模与 ASR 对比数字。  
筛选脚本：`filter_tier_A.py`。

---

## 1. 结论速览

**L 全量（10% + 剩余 90%）的 ASR 推理与三方一致筛选均已完成。**

| 池子 | ASR 推理 | 三方一致筛选 | Tier-A 规模 |
|------|----------|--------------|-------------|
| L **10%** 固定抽样 | 已完成 | **已完成** | **1,015,329 句 / 619.1 h** |
| L **剩余 ~90%** | 已完成 | **已完成**（2026-11-09 16:43） | **9,141,237 句 / 5,571.6 h** |
| **合计** | — | — | **10,156,566 句 / ~6,190.7 h** |

一致率非常稳定：两个池子都约 **69.4% 条数 / 61.9% 时长** 通过三方一致。

---

## 2. 三方一致结果保存在哪里

> **最常用入口**：[`tier_A_10shards/`](tier_A_10shards/) —— L 全量 1,015.7 万句 / 6,190.7 h 已按时长均衡切成 **10 份**，
> 每份 ≈101.5 万句 / 619.07 h / 416 MB。详见 [2.4](#24-十份分片推荐使用)。

### 2.1 主产物（本机工作区，含绝对音频路径）

| 文件 | 说明 |
|------|------|
| [`outputs/tier_A_agree3.jsonl`](outputs/tier_A_agree3.jsonl) | **主清单**，1,015,329 行；`audio_path` 为本机绝对路径 |
| [`outputs/tier_A_summary.json`](outputs/tier_A_summary.json) | 筛选统计（条数、小时、来源、confidence） |
| [`outputs/tier_A_agree3_sample100.jsonl`](outputs/tier_A_agree3_sample100.jsonl) | 随机 100 条预览 |

### 2.2 开源整理包（相对路径，便于分发）

| 路径 | 说明 |
|------|------|
| [`tier_A_open/`](tier_A_open/) | 独立开源目录（README / LICENSE / CITATION） |
| [`tier_A_open/data/tier_A_agree3.jsonl`](tier_A_open/data/tier_A_agree3.jsonl) | 与主清单同规模；`audio_path` 为相对 WenetSpeech 根目录 |
| [`tier_A_open/data/meta.json`](tier_A_open/data/meta.json) | 元数据 |
| [`tier_A_open/data/samples_100.jsonl`](tier_A_open/data/samples_100.jsonl) | 预览样例 |
| [`tier_A_open/scripts/extract_wav.py`](tier_A_open/scripts/extract_wav.py) | 按清单切 wav |

两份 JSONL **条数相同**（1,015,329），差异主要是 `audio_path` 绝对 vs 相对。日常训练/听测用 `outputs/`；对外发布用 `tier_A_open/`。

### 2.3 剩余 90%（已生成）

| 文件 | 说明 |
|------|------|
| [`outputs/tier_A_agree3_rest90.jsonl`](outputs/tier_A_agree3_rest90.jsonl) | **主清单**，9,141,237 行 / 3.7 GB；`audio_path` 绝对路径 |
| [`outputs/tier_A_summary_rest90.json`](outputs/tier_A_summary_rest90.json) | 筛选统计 |

完整性已校验：9,141,237 行、`utt_id` 全部唯一、无解析失败行。

生成命令（后台模式，日志 `outputs/tier_A_filter_rest90.log`）：

```bash
cd /cmy/cmy/ranAudio/wenetSpeechL
bash launch_rest90.sh filter
bash launch_rest90.sh status   # 看进度
```

与 10% 合并开源时，可拼接：

- `tier_A_open/data/tier_A_agree3.jsonl`（1,015,329）
- `outputs/tier_A_agree3_rest90.jsonl`（9,141,237）

> 注：rest90 的 `audio_path` 是绝对路径，发布前需按 `tier_A_open` 的做法转成相对路径。

### 2.4 十份分片（推荐使用）

目录：[`tier_A_10shards/`](tier_A_10shards/)

把 **L10 + rest90 两份合并后的全量**（10,156,566 句 / 6,190.72 h）按时长均衡切成 10 份：

| 项 | 数值 |
|----|------|
| 分片 | `part_00.jsonl` … `part_09.jsonl` |
| 每片句数 | 1,014,920 – 1,016,560（≈101.5 万） |
| 每片时长 | 619.0715 – 619.0723 h（误差 < 10 s） |
| 每片大小 | 约 376 MB（合计 ~3.8 GB） |
| 每片来源 | podcast / youtube ≈ 50 / 50 |
| 每片池子 | L10 约 10.1 万 + rest90 约 91.4 万 |
| 每片 audio_path | **相对路径**（`audio/train/...`） |
| 元数据 | [`manifest.json`](tier_A_10shards/manifest.json)、[`README.md`](tier_A_10shards/README.md) |

切分规则：贪心最小堆积 —— 每条样本分配给「当前累计时长最少」的分片。

校验：10,156,566 行、`utt_id` 全局唯一（0 重复 / 0 坏行）。

生成脚本：[`split_tier_A_10shards.py`](split_tier_A_10shards.py)

```bash
python split_tier_A_10shards.py --n 10 --out-dir tier_A_10shards
```

---

## 3. Tier-A（L 10%）详细统计

来源：`outputs/tier_A_summary.json` / `tier_A_open/data/meta.json`。

| 项 | 数值 |
|----|------|
| 父池 | `sample_manifest.jsonl`：1,462,257 句 / ~999.9 h |
| Tier-A 句数 | **1,015,329** |
| Tier-A 时长 | **~619.13 h** |
| 占父池条数 | **69.44%** |
| 占父池时长 | **61.92%** |

### 按来源

| source | 句数 | 小时 | 占 Tier-A 条数 |
|--------|------|------|----------------|
| podcast | 506,368 | 293.71 | 49.87% |
| youtube | 508,961 | 325.42 | 50.13% |

### 按官方 confidence

| 区间 | 句数 | 小时 | 占比 |
|------|------|------|------|
| `1.0` | 1,013,081 | 615.48 | 99.78% |
| `[0.95, 1)` | 2,248 | 3.65 | 0.22% |

### 字段（每行一条 JSON）

| 字段 | 说明 |
|------|------|
| `utt_id` | WenetSpeech segment id |
| `text` | 官方原始文本（推荐作监督标签） |
| `text_norm` | 去标点后规范文本（三方一致所用） |
| `hyp_qwen` / `hyp_firered` | 两路 ASR 原始输出 |
| `confidence` | 官方强制对齐 confidence |
| `source` | `podcast` / `youtube` |
| `start` / `duration` | 切段秒级区间 |
| `audio_path` | 音频路径（绝对或相对，见上） |
| `tier` | 固定 `"A"` |

一致规则：`strip_punctuation` 后  
`norm(官方 text) == norm(hyp_qwen) == norm(hyp_firered)`。

### 3.2 Tier-A（L 剩余 ~90%）详细统计

来源：`outputs/tier_A_summary_rest90.json`。

| 项 | 数值 |
|----|------|
| 父池 | `sample_manifest_rest90.jsonl`：13,162,970 句 / ~9,005.2 h |
| Tier-A 句数 | **9,141,237** |
| Tier-A 时长 | **~5,571.59 h** |
| 占父池条数 | **69.45%** |
| 占父池时长 | **61.87%** |

按来源：

| source | 句数 | 小时 | 占 Tier-A 条数 |
|--------|------|------|----------------|
| podcast | 4,552,483 | 2,638.59 | 49.80% |
| youtube | 4,588,754 | 2,933.00 | 50.20% |

按官方 confidence：

| 区间 | 句数 | 小时 | 占比 |
|------|------|------|------|
| `1.0` | 9,121,144 | 5,538.40 | 99.78% |
| `[0.95, 1)` | 20,093 | 33.19 | 0.22% |

逐千万进度（`tier_A` 累计）：1,000 万句时 679.9 万通过；1,300 万句时 902.2 万；最终 914.1 万。

### 3.3 L 10% + 90% 合并

| 项 | 数值 |
|----|------|
| 总句数 | **10,156,566** |
| 总时长 | **~6,190.72 h** |
| 一致率（条数） | 69.4% / 69.5%（两池几乎一致） |
| 一致率（时长） | 61.9% / 61.9% |

---

## 4. ASR 对比结果（相对官方标签 CER）

CER 来自各目录 `last_run_summary.json` 的 `global_cer`（按字符错误累计；标点未完全对齐到 Tier-A 的规范化，仅作模型相对强弱参考）。

### 4.1 L 10%（`sample_manifest.jsonl`，1,462,257 句）

| 模型 | 输出目录 | ok | fail | global CER | 本轮耗时 |
|------|----------|----|------|------------|----------|
| Qwen3-ASR-1.7B | `outputs/qwen3_asr_L10pct/`（30 shards） | 1,462,257 | 0 | **4.70%** | ~43.7 h |
| FireRedASR2-AED | `outputs/firered_asr_L10pct/`（31 shards） | 1,462,057 | 0 | **3.13%** | ~81.7 h |

FireRed 少约 200 条 ok，属切片/解码边界差异；Tier-A 取两路与官方的交集，`n_common = 1,462,257`。

### 4.2 L 剩余 ~90%（`sample_manifest_rest90.jsonl`，13,162,970 句）

| 模型 | 输出目录 | 末次 summary | global CER（末次累计段） | 说明 |
|------|----------|--------------|--------------------------|------|
| Qwen3-ASR-1.7B | `outputs/qwen3_asr_L90pct/`（408 shards） | ok=7,145,370；`already_done_before`=6,017,600 | **4.48%** | 含失败行重试；shard 可能对同 `utt_id` 有多行，**以最新为准** |
| FireRedASR2-AED | `outputs/firered_asr_L90pct/`（264 shards） | ok=7,292,570；`already_done_before`=5,870,400 | **3.00%** | 推理已完成 |

末次 `elapsed_sec`：Qwen ~94.6 h，FireRed ~172.0 h（仅该轮；全量含此前续跑更长）。

要点：

- **FireRed 更准**（CER 约低 1.5 个点），**Qwen 更快**。  
- L10 / L90 上两模型相对关系一致。  
- L90 `filter` 已完成：1,316 万共同句中 **914.1 万条（69.45%）/ 5,571.6 h** 通过三方一致。

### 4.3 rest90 筛选的实际表现

| 项 | 数值 |
|----|------|
| Qwen 索引（SQLite）| 13,162,970 条，临时库 ~4.5 GB |
| FireRed 扫描 | 13,162,970 条 |
| 共同句 | 13,162,970（两路无缺失） |
| 通过三方一致 | 9,141,237（69.45%） |
| 输出 | 3.7 GB JSONL |
| 耗时 | 约 2.5 h（2026-11-09 14:45 → 16:43，纯 CPU） |
| 内存 | 峰值 RSS 约 50 MB（流式，不再 OOM） |

早期版本把 Qwen/FireRed 两侧全量读进 dict，在 32 GiB 容器 cgroup 限制下被 OOM kill；改成 SQLite 索引 + 流式比较后一次跑通。

---

## 5. 相关输入与中间产物路径

```text
# 划分清单
wenetSpeechL/sample_manifest.jsonl          # L 10%（1,462,257）
wenetSpeechL/sample_manifest_rest90.jsonl   # L 补集 ~90%（13,162,970）

# 原始音频 / 官方标注
/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/
/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/WenetSpeech.json

# 模型
/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B
/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED

# 推理 shards
wenetSpeechL/outputs/qwen3_asr_L10pct/
wenetSpeechL/outputs/firered_asr_L10pct/
wenetSpeechL/outputs/qwen3_asr_L90pct/
wenetSpeechL/outputs/firered_asr_L90pct/

# Tier-A（10% + rest90 均已完成）
wenetSpeechL/outputs/tier_A_agree3.jsonl
wenetSpeechL/outputs/tier_A_summary.json
wenetSpeechL/outputs/tier_A_agree3_rest90.jsonl
wenetSpeechL/outputs/tier_A_summary_rest90.json
wenetSpeechL/tier_A_open/data/tier_A_agree3.jsonl
```

---

## 6. 复现 / 续跑命令

```bash
cd /cmy/cmy/ranAudio/wenetSpeechL

# --- L 10%（已完成，仅复现时）---
bash launch_bg.sh qwen
bash launch_bg.sh firered
python filter_tier_A.py \
  --qwen_dir outputs/qwen3_asr_L10pct \
  --firered_dir outputs/firered_asr_L10pct \
  --out outputs/tier_A_agree3.jsonl \
  --summary outputs/tier_A_summary.json

# --- L ~90%：推理与 Tier-A 筛选均已完成（仅复现时）---
bash launch_rest90.sh filter    # 后台，日志 outputs/tier_A_filter_rest90.log
bash launch_rest90.sh status
```

> `filter_tier_A.py` 现在用 **SQLite 做 Qwen 侧索引**（`outputs/.tier_a_qwen_*.sqlite`，完成后自动删除），
> 避免 L90 约 1300 万条双边 dict 在 32 GiB 容器内存里被 OOM 杀掉；顺带实现了流式比较。
> rest90 全量筛选耗时约 **2.5 h**（CPU/磁盘密集，不用 GPU）。

环境：`/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python`（与 10% 相同）。

---

## 7. 文档索引

| 文档 | 内容 |
|------|------|
| 本文 `TIER_A_RESULTS.md` | 三方一致落盘位置 + Qwen/FireRed 结果数字 |
| [`tier_A_open/README.md`](tier_A_open/README.md) | 开源包字段、切音频、许可 |
| [`REST90.md`](REST90.md) | 剩余 90% 流水线 |
| [`README.md`](README.md) §8 | 数据集总说明中的对比入口 |

---

## 8. 免责声明

- 官方 L 标签来自 OCR / 商用 ASR + CTC confidence，**不是**人工金标。  
- Tier-A 是自动化多路共识，降低噪声，**不保证**每条绝对正确；关键用途建议抽听。  
- 音频版权与使用遵循 WenetSpeech 官方条款；本仓库不附带音频。
