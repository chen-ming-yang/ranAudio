# WenetSpeech L 数据集说明

本文档描述本机上的 **WenetSpeech L（约 10000 小时）** 中文 ASR 训练集：原始音频位置、子集含义、以及 real-asr 中已转换的 Lhotse manifest。

官方简介：WenetSpeech 是面向中文语音识别的大规模开源数据集（约 10000+ 小时，CC-BY-4.0），原始标注见 `WenetSpeech.json`。

论文：[WenetSpeech: A 10000+ Hours Multi-domain Mandarin Corpus for Speech Recognition](https://arxiv.org/abs/2110.03370)（ICASSP 2022）。

---

## 0. L 集文本是怎么来的？（结合论文）

**结论：L 不是用某一个开源 ASR 模型整集转写出来的。**  
文本候选来自两条管道（按数据源），再用另一套 **CTC 端到端强制对齐** 打 confidence，筛出 Strong Label；**L = 全部 Strong Label（confidence ∈ [0.95, 1.0]，约 10005h）**。

### 0.1 候选文本生成（源相关）

| 数据源 | High Label 约小时 | 文本候选怎么来 | 论文中的模型/系统 |
|--------|-------------------|----------------|-------------------|
| **YouTube** | ~6113h | 对视频**硬字幕区域做 OCR**，按字幕起止切音文对 | **CRNN-CTC** 文本识别（先检字幕框，再识别）；作者报告在 5k 人工核对字幕上识别准确率约 **98%** |
| **Podcast** | ~3892h | **第三方商业 ASR** 转写 | 论文写作 *“a third-party commercial ASR transcription system”*，称其在公开评测平台上表现顶尖；先 **VAD** 切段再识别。**未公开具体厂商/模型名** |

要点：

- YouTube 侧本质是「读屏上字幕」，不是听音频的 ASR；字幕本身可能是人工或平台字幕，OCR 只是把它变成文本并对齐时间。
- Podcast 侧才是「ASR 生成文本」；但具体是哪家商用引擎，论文与官方 README **均未点名**。
- 论文明确写：转写系统与后续校验用的 force-alignment 模型，**训练方法与数据均不同**（避免同源自洽虚高）。

### 0.2 标签校验与 confidence（筛出 L）

对上述候选统一做 **CTC-based end-to-end force alignment**（WeNet 工具链实现）：

1. 按 CTC 建模单元切分参考文本，构图允许插入/替换等到 filler；
2. 与 CTC topology 合成强制解码图 `F = T ∘ L`；
3. 用参考文本 (ref) 与强制解码假设 (hyp) 算 **confidence**（标签–音频一致程度）；
4. 按 confidence 分箱：

| 集合 | Confidence | 小时 | 用途 |
|------|------------|------|------|
| Strong Label（**含训练子集 L**） | [0.95, 1.0] | 10005 | 有监督训练 |
| Weak Label | [0.60, 0.95) | 2478 | 半监督/噪声训练 |
| Others / Unlabel | &lt;0.6 或无效 | ~9952 | 无监督等 |

**S / M**：从 **confidence = 1.0** 的训练数据中抽样（100h / 1000h）。  
**L**：几乎等于全部 Strong Label，因此仍可能含 OCR 错、字幕错、ASR 词错等，只是校验分数 ≥ 0.95。

### 0.3 和「训练 baseline 用的模型」区分

论文 Section 4 的 Kaldi chain / ESPnet Conformer / WeNet 等，是 **在 WenetSpeech 上训出来的 baseline**，用来报 Dev / Test_Net / Test_Meeting 指标，**不是**用来生成 L 标签的模型。

### 0.4 对本机数据的含义

`supervisions.jsonl` 里的 `text` + `custom.confidence` + `custom.subsets` 即上述流水线产物：

- `subsets` 含 `"L"`：落入 Strong Label 训练子集；
- `confidence` 接近 1.0：强制对齐认为音文很吻合（S/M 抽样池）；略低但仍 ≥0.95：仍在 L 内，噪声可能略多。

---

## 1. 本机路径总览

| 内容 | 路径 | 说明 |
|------|------|------|
| 原始根目录 | `/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/` | 官方解压后的数据根 |
| 标注 JSON | `.../WenetSpeech.json` | 约 6.2G，含全部 audios / segments |
| 音频目录 | `.../audio/` | 约 **310G**，主要是 `.opus` |
| L 训练 manifest | `/cmy/cmy/real-asr/data/manifests/wenetspeech/train/` | Lhotse recordings + supervisions |
| M+S 训练 manifest | `/cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms/` | 高质量约 1000h 子集 |
| 本说明目录 | `/cmy/cmy/ranAudio/wenetSpeechL/` | 仅文档，不含音频副本 |

音频按 split 分子目录：

```
audio/
  train/          # 训练长音频（YouTube 等）
  dev/
  test_net/
  test_meeting/
```

文件格式多为 **Opus 编解码的语音文件**（`.opus`），不是模型名。采样率在 Lhotse manifest 中记为 **16 kHz**。

---

## 2. S / M / L 子集区别

三者是**同一份语料上的嵌套子集**（`S ⊂ M ⊂ L`），按规模与标注可靠度划分，不是不同域的独立数据集。

| 子集 | 约小时数 | 特点 | 本仓库常用场景 |
|------|----------|------|----------------|
| **S** | ~100h | 最高质量小集 | 快速实验 / 含在 M+S 中 |
| **M** | ~1000h | 人工校验或更可靠标注 | 阶段训练、fbank 预计算 |
| **L** | ~10000h | 含大量弱标签/自动标注 | 大规模预训练 |

`WenetSpeech.json` 里每条 segment 有 `subsets` 字段（如 `["L"]` 或 `["S","M","L"]`）。准备数据时按该字段过滤。

---

## 3. 本机已生成的 Lhotse 规模

由 `real-asr/data/prepare_wenetspeech.py` 从官方 JSON 流式转换而来。

| Split / 子集 | recordings | supervisions | 备注 |
|--------------|------------|--------------|------|
| **train (L)** | 61,127 | **14,625,227** | ~10000h 训练 |
| **train_ms (M+S)** | 58,555 | 1,514,497 | ~1000h |
| **dev** | 13,825 | 13,825 | 验证 |
| **test_net** | 124 | 24,774 | 网络场景测试 |
| **test_meeting** | 8,370 | 8,370 | 会议场景测试 |

磁盘占用（manifest）：

- L train：约 **4.0G**
- M+S train：约 **420M**

### 示例（L train）

Recording（长音频一条）：

```json
{
  "id": "Y0000000000_--5llN02F84",
  "sources": [{"type": "file", "channels": [0],
    "source": "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus"}],
  "sampling_rate": 16000,
  "duration": 2494.57
}
```

Supervision（切分段一条）：

```json
{
  "id": "Y0000000000_--5llN02F84_S00000",
  "recording_id": "Y0000000000_--5llN02F84",
  "start": 20.08,
  "duration": 4.32,
  "text": "怎么样这些日子住得还习惯吧",
  "language": "zh",
  "custom": {"confidence": 1.0, "subsets": ["L"]}
}
```

转换时常见过滤：空文本丢弃；时长限制约 **0.3s ~ 30s**。

---

## 4. 如何重新生成 manifest

在 `real-asr` 目录下：

```bash
cd /cmy/cmy/real-asr

# 生成 L（默认 subsets=L）
python data/prepare_wenetspeech.py \
  --wenetspeech-json /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/WenetSpeech.json \
  --audio-root /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY \
  --output-dir data/manifests/wenetspeech \
  --subsets L \
  --splits train dev test_net test_meeting

# 生成 M（或 S M 等）
python data/prepare_wenetspeech.py \
  --subsets M S \
  --output-dir data/manifests/wenetspeech_ms_tmp
```

当前仓库里 L 输出在 `data/manifests/wenetspeech/train/`，M+S 在 `train_ms/`。

---

## 5. 训练侧用法（real-asr）

| 用途 | 配置 / 脚本 | 数据 |
|------|-------------|------|
| CTC 大规模预训练 | `configs/train_ctc_L.yaml`、`train_ctc_L.sh` | L train manifest |
| RNN-T L | `train_rnnt/config_L.yaml`、`run_train_rnnt_L.sh` | 同上 |
| 日常阶段训 / fbank shard | `train_stage1*.yaml`、`run_precompute_fbank.sh` | **train_ms**（M+S） |

预计算 fbank（MS）：`/cmy/cmy/real-asr/data/fbank_shards/wenetspeech_ms/`（约 1513 shards）。完整 L 训练多走**在线读 opus + 提特征**，直接用 L manifest。

---

## 6. 使用注意

1. **不要把 310G 音频复制进本目录**；文档只做索引，音频以 `DATA_UNTAR_DIRECTORY` 为准。
2. L 含弱标签，`custom.confidence` 可作过滤参考；要干净基线优先用 **M/S**。
3. Opus 解码依赖系统/环境的 `libopus` / `torchaudio` / `ffmpeg` 等；训练前确认能正常读 `.opus`。
4. 官方声明：WenetSpeech 不拥有全部音频版权；按 `TERMS_OF_ACCESS` / 官方说明使用。

---

## 7. 相关引用路径速查

```
原始音频:     /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/
官方标注:     /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/WenetSpeech.json
L 列表:       /cmy/cmy/real-asr/data/manifests/wenetspeech/train/
M+S 列表:     /cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms/
准备脚本:     /cmy/cmy/real-asr/data/prepare_wenetspeech.py
本说明:       /cmy/cmy/ranAudio/wenetSpeechL/README.md
```

---

## 8. Qwen3-ASR / FireRedASR2 重转写对比（L 10%）

对 L 的 **10% 固定样本**（`sample_manifest.jsonl`，约 146 万句）分别用 Qwen3-ASR-1.7B 与 FireRedASR2-AED 重跑，与官方 `text` 算 CER。

### 环境（复用已有，不重复装）

| 用途 | 环境 |
|------|------|
| Qwen / 推理 | `wenet-asr-compare`（自 `qwen3-asr` 克隆，transformers==4.57.6）或原 `qwen3-compare` |
| FireRed 权重 | ModelScope 镜像 `xukaituo/FireRedASR2-AED` → `/cmy/cmy/FireRedASR2S/pretrained_models/` |
| 模型权重 Qwen | 本机已有 `/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B`（不重下） |

### 命令

```bash
# smoke（已跑通：200 条 gCER≈0.046）
bash wenetSpeechL/_start_qwen_smoke.sh

# 全量 10% 后台（单卡串行：先 Qwen，AED 下完后再 firered）
bash wenetSpeechL/launch_bg.sh qwen
# AED 就绪后：
bash wenetSpeechL/launch_bg.sh firered

# 监控
tail -f wenetSpeechL/outputs/qwen3_asr_L10pct/run.log
wc -l wenetSpeechL/outputs/qwen3_asr_L10pct/shard_*.jsonl

# 汇总
python wenetSpeechL/summarize_compare.py
```

### Tier-A 开源包与结果落盘

三方一致（官方 = Qwen = FireRed，去标点）子集已整理；**详细数字与路径见 [`TIER_A_RESULTS.md`](TIER_A_RESULTS.md)**。

- **十份分片（推荐）**：[`tier_A_10shards/`](tier_A_10shards/) —— 全量 1,015.7 万句 / 6,190.7 h，每份 ≈101.5 万句 / 619.07 h
- 本机合并前主清单：`outputs/tier_A_agree3.jsonl`（L10）、`outputs/tier_A_agree3_rest90.jsonl`（rest90）
- 开源包（仅 L10，相对路径）：[`tier_A_open/`](tier_A_open/)

### 剩余 ~90%（已完成）

与 10% 相同流程；**Qwen / FireRed 推理 + Tier-A 筛选全部完成**：

- `outputs/tier_A_agree3_rest90.jsonl`：**914.1 万句 / 5,571.6 h**（一致率 69.45%）
- 与 10% 合计：**1,015.7 万句 / ~6,190.7 h** 三方一致
- 结果与对比：[`TIER_A_RESULTS.md`](TIER_A_RESULTS.md)
- 流水线：[`REST90.md`](REST90.md)
- 重跑/续跑：`bash launch_rest90.sh filter`（后台）
- 进度：`bash launch_rest90.sh status`
