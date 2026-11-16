# WenetSpeech-L Tier-A Agree3

高一致性中文语音–文本子集：在 **WenetSpeech L 的 10% 抽样** 上，要求

**官方标签 = Qwen3-ASR-1.7B = FireRedASR2-AED**

（去中英文标点与空白后比较）。可作为准金标监督 / 评测筛选池。

本机完整结果路径、L10/L90 ASR CER、rest90 筛选状态见上级文档：[`../TIER_A_RESULTS.md`](../TIER_A_RESULTS.md)。  
工作区绝对路径副本：`../outputs/tier_A_agree3.jsonl`（与本目录 `data/tier_A_agree3.jsonl` 同规模）。

> **本仓库只发布标注清单（JSONL），不包含音频。**  
> 音频请自行获取 [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech) 原始数据，并按 `audio_path` + `start`/`duration` 切段。

---

## 1. 规模

| 项 | 数值 |
|----|------|
| 语句数 | **1,015,329** |
| 时长 | **~619.1 小时** |
| 父池 | WenetSpeech L train 的 10% 抽样（1,462,257 句 / ~999.9 h） |
| 占父池条数 | **69.4%** |
| 占父池时长 | **61.9%** |

按来源：

| source | 句数 | 小时 |
|--------|------|------|
| podcast | 506,368 | 293.7 |
| youtube | 508,961 | 325.4 |

官方 `confidence`：约 **99.8%** 为 `1.0`，其余为 `[0.95, 1)`。

详见 [`data/meta.json`](data/meta.json)。

---

## 2. 目录结构

```text
tier_A_open/
  README.md                 # 本说明
  LICENSE                   # 许可与归属
  CITATION.md               # 引用
  data/
    tier_A_agree3.jsonl     # 主清单（相对路径）
    samples_100.jsonl       # 随机 100 条预览
    meta.json               # 统计元数据
  scripts/
    extract_wav.py          # 按清单切 wav
```

---

## 3. 字段说明

每行一条 JSON（UTF-8 JSONL）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `utt_id` | str | WenetSpeech segment id |
| `text` | str | 官方原始文本（推荐作监督标签） |
| `text_norm` | str | 去标点/空白后的规范文本（三方一致所用） |
| `hyp_qwen` | str | Qwen3-ASR-1.7B 原始输出 |
| `hyp_firered` | str | FireRedASR2-AED 原始输出 |
| `confidence` | float | 官方 WenetSpeech 强制对齐 confidence |
| `source` | str | `podcast` / `youtube` |
| `start` | float | 在长音频中的起点（秒） |
| `duration` | float | 时长（秒） |
| `audio_path` | str | **相对** WenetSpeech 根目录的路径，如 `audio/train/podcast/.../xxx.opus` |
| `tier` | str | 固定为 `"A"` |

示例：

```json
{
  "utt_id": "X0000010828_94000021_S00636",
  "text": "其实中国的传统媒体的商业化是跟互联网媒体同时发生的",
  "text_norm": "其实中国的传统媒体的商业化是跟互联网媒体同时发生的",
  "hyp_qwen": "其实，中国的传统媒体的商业化是跟互联网媒体同时发生的。",
  "hyp_firered": "其实中国的传统媒体的商业化是跟互联网媒体同时发生的",
  "confidence": 1.0,
  "source": "podcast",
  "start": 1345.55,
  "duration": 4.4,
  "audio_path": "audio/train/podcast/B00042/X0000010828_94000021.opus",
  "tier": "A"
}
```

完整音频路径：

```text
<WENETSPEECH_ROOT>/<audio_path>
```

切段区间：`[start, start + duration)`（秒）。

---

## 4. 如何构造（方法）

1. 从 WenetSpeech L 训练监督中按 utt_id 哈希抽取约 **10%**。  
2. 用 **Qwen3-ASR-1.7B** 与 **FireRedASR2-AED** 对每段重转写。  
3. 对官方 `text`、两路 hyp 做相同规范化：去掉常见中英文标点与空白。  
4. 仅保留三方规范文本 **完全相等** 的条目 → Tier A。

说明：官方 L 标签本身来自 OCR / 商用 ASR + CTC confidence 过滤（见 WenetSpeech 论文），**不是**人工金标准。Tier A 通过「双强模型 + 官方」共识降低噪声，仍建议在关键应用前做小规模听测。

---

## 5. 切音频示例

```bash
# 需要本机已解压 WenetSpeech，且安装 ffmpeg
python scripts/extract_wav.py \
  --jsonl data/samples_100.jsonl \
  --audio-root /path/to/WenetSpeech/DATA_UNTAR_DIRECTORY \
  --out-dir ./wav_preview \
  --max 10
```

单条手动切：

```bash
ffmpeg -ss 1345.55 -t 4.4 \
  -i $WENETSPEECH_ROOT/audio/train/podcast/B00042/X0000010828_94000021.opus \
  -ac 1 -ar 16000 out.wav
```

---

## 6. 许可与归属

- **本清单**：见 [LICENSE](LICENSE)。构建过程产生的筛选结果以可再分发条款发布；使用时须同时遵守 WenetSpeech 与相关模型许可。  
- **音频与原始标注**：归 WenetSpeech（CC-BY-4.0 等，以官方为准），本仓库不附带音频。  
- **模型**：Qwen3-ASR、FireRedASR2 请遵循各自官方许可。

引用见 [CITATION.md](CITATION.md)。

---

## 7. 免责声明

Tier A 是自动化多路一致性筛选结果，**不能保证**每条标签绝对正确。开源用途包括研究、训练数据清洗与基准构建；请勿将「共识」误当作人工校对金标。
