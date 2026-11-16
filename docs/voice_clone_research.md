# 构音障碍音色克隆 — 调研与使用文档

> 日期: 2026-11-03（2026-11-05 修订同步最终结论）
> 试点: S06 说话人（MDSD 数据集，9,278 条，6.26h）
> 状态: S06 试点全流程完成。零样本 xvec 为最终方案；微调路线已验证失败、弃用

---

## 1. 背景与目标

构音障碍（dysarthria）患者因神经肌肉控制受损，发音不清晰，导致语音可懂度低。本方案利用 **Qwen3-TTS voice clone** 能力，以患者录音为参考克隆其音色，输入新文本生成**清晰语音**，帮助患者用"自己的声音"说清楚话。

**核心思路**：Qwen3-TTS 预训练模型天然生成清晰语音。speaker encoder 从构音障碍参考音频中提取说话人音色向量（x-vector），generator 以新文本为条件生成清晰语音——保留音色、输出清晰。

---

## 2. 技术方案

### 2.1 技术路线

```
构音障碍录音 (ref_audio) ──→ Speaker Encoder ──→ 音色向量 (x-vector)
                                                        │
新文本 (text) ──→ Text Encoder ──→ 条件 ──→ Generator ──→ 清晰语音
```

### 2.2 两种克隆模式

| 模式 | API 参数 | 原理 | 适用场景 |
|------|---------|------|---------|
| **xvec** | `x_vector_only_mode=True` | 仅提取说话人向量 | **推荐**——不迁移构音韵律 |
| **ICL** | `x_vector_only_mode=False` | 上下文克隆（ref_text + ref_code） | 参考音频质量高时 |

### 2.3 零样本 vs 微调

- **零样本**：直接用构音障碍音频做 voice clone，利用预训练模型的清晰语音先验。**已验证有效，为最终方案**。
- **微调**：对特定说话人 SFT，期望提升音色相似度（需将音频重采样到 24kHz）。**已验证失败并弃用**——模型学到了构音发音缺陷，CER 始终接近 1.0，详见 §5.4。流程保留仅供复现。

---

## 3. 环境配置

| 组件 | 路径/环境 |
|------|----------|
| Qwen3-TTS 推理/微调 | `conda: qwen3-tts` (`/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python`) |
| Qwen3-ASR 评估 | `conda: wenet-asr-compare` + `PYTHONPATH=/cmy/cmy/Qwen3-ASR` |
| 说话人验证 (cam++) | `conda: wenet-asr-compare`（已装 funasr）或 `conda: funasr` |
| 音频重采样 (16k→24k) | `conda: funasr` |
| 数据索引 | `data/dysarthric.jsonl` |
| 音频数据 | `/cmy/data/after_catting/10h/Audio/06/*.wav` |

```bash
export PY_TTS=/cmy/cmy/miniconda3/envs/qwen3-tts/bin/python
export PY_EVAL=/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python
cd /cmy/cmy/ranAudio
```

---

## 4. 完整 Pipeline

### Step 1: 数据准备

```bash
# 解析数据、选参考音频、划分数据集
$PY_TTS -m ranAudio.voice_clone.data_prep --spk 06

# 重采样到 24kHz（微调需要）
/cmy/cmy/miniconda3/envs/funasr/bin/python -m ranAudio.voice_clone.resample --spk 06
```

产物: `runs/voice_clone/S06/{finetune_train.jsonl, test_texts.jsonl, ref/}`

### Step 2: 零样本克隆

```bash
# xvec 模式（推荐）
$PY_TTS -m ranAudio.voice_clone.zeroshot --spk 06 --from-test 5 --mode xvec

# ICL 模式
$PY_TTS -m ranAudio.voice_clone.zeroshot --spk 06 --from-test 5 --mode icl
```

产物: `runs/voice_clone/S06/zeroshot_xvec/` 或 `zeroshot_icl/`

### Step 3: 微调（已弃用，仅保留供复现）

> ⚠️ 该路线已验证失败（CER≈1.0，详见 §5.4），**不要用于生产**。以下步骤仅供复现实验。
> 复现前需确认 `/cmy/cmy/Qwen3-TTS/finetuning/sft_12hz.py` 已补上 `text_projection`（§5.4 问题 1），
> 并使用 `sft_wrapper.py`（自动降级 flash-attn）。`--speaker_name` 必须用**小写** `s06`。

```bash
# 提取 audio_codes
$PY_TTS /cmy/cmy/Qwen3-TTS/finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path /cmy/cmy/Qwen3-TTS/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl runs/voice_clone/S06/finetune_train_24k.jsonl \
  --output_jsonl runs/voice_clone/S06/finetune_train_24k_with_codes.jsonl

# 执行 SFT
cd /cmy/cmy/Qwen3-TTS/finetuning
$PY_TTS sft_wrapper.py \
  --init_model_path /cmy/cmy/Qwen3-TTS/Qwen3-TTS-12Hz-0.6B-Base \
  --output_model_path /cmy/cmy/ranAudio/runs/voice_clone/S06/checkpoint \
  --train_jsonl /cmy/cmy/ranAudio/runs/voice_clone/S06/finetune_train_24k_with_codes.jsonl \
  --batch_size 2 --lr 2e-6 --num_epochs 5 --speaker_name s06
```

### Step 4: 效果评估

```bash
PYTHONPATH=/cmy/cmy/Qwen3-ASR $PY_EVAL -m ranAudio.voice_clone.evaluate \
  --spk 06 --modes xvec --max-items 50
```

产物: `runs/voice_clone/S06/eval_report.md`（由 `generate_report()` 自动覆盖写入）。
当前文件末尾另有**手工补充**的三方案综合对比章节（§5.3/5.4 结论），重跑评估会覆盖该章节，如需保留请重新追加。

### Step 5: 推理 Demo

```bash
# 单条推理
$PY_TTS -m ranAudio.voice_clone.demo --spk 06 --text "今天天气真好" --mode xvec

# 批量推理
$PY_TTS -m ranAudio.voice_clone.demo --spk 06 --text "你好" "我们一起去看电影" --mode xvec

# 微调模型推理（已弃用，仅供效果对比）
$PY_TTS -m ranAudio.voice_clone.demo --spk 06 --text "你好" --mode finetuned
```

---

## 5. 试点结果（S06，5 条测试）

### 5.1 零样本 xvec 模式

| 指标 | 克隆语音 | 原始构音语音 |
|------|---------|------------|
| **CER (Qwen3-ASR)** | **0.20** | **1.22** |
| 完美识别率 | 4/5 (80%) | 0/5 (0%) |

**结论**: 克隆语音可懂度远超原始构音语音，方案有效。

### 5.2 详细对比

| 目标文本 | 克隆 ASR 转写 | 克隆 CER | 原始 CER |
|---------|-------------|---------|---------|
| 齐 | 其。 | 1.000 | 1.000 |
| 一旦发生 | 一旦发生。 | 0.000 | 0.750 |
| 爸爸 | 爸爸。 | 0.000 | 1.000 |
| 马虎的 | 马虎的。 | 0.000 | 2.333 |
| 大家一起努力吧 | 大家一起努力吧！ | 0.000 | 1.000 |

### 5.3 三方案综合对比（最终结论）

| 方案 | CER↓ | 说话人相似度↑ | 推理耗时 | 结论 |
|------|------|-------------|---------|------|
| **零样本 xvec** | **0.20** | **0.49** | ~6s/条 | ✅ **推荐，最终方案** |
| 零样本 ICL | 1.40 | 0.35 | ~350s/条 | ❌ 迁移了构音特征 |
| 微调 (epoch-4) | 0.92 | 0.16 | 1-2s/条 | ❌ 清晰度被破坏 |
| 原始构音语音（基线） | 1.22 | 0.70 | — | 不可懂 |

**说话人相似度**（cam++ cosine，与参考音频对比）：
- 原始构音语音 0.70（同说话人上限参考）
- 零样本 xvec 0.49
- 零样本 ICL 0.35

### 5.4 微调方案验证：失败，弃用

训练完整跑完 5 个 epoch（3000 条，loss 11.6 → 5.0，checkpoint 齐全）。按 epoch 评估：

| checkpoint | CER | 说话人相似度 |
|-----------|-----|------------|
| epoch-0 | 1.00 | 0.14 |
| epoch-2 | 0.88 | 0.25 |
| epoch-4 | 1.00 | 0.29 |

**结论**：随训练推进相似度缓慢上升（确实在学音色），但 **CER 始终接近 1.0，清晰度完全丧失**——模型学到了构音语音的发音缺陷，破坏了预训练的清晰语音先验。这与方案预判的风险一致，故**微调路线不采用**，以零样本 xvec 为最终方案。

**过程中的三个工程问题（已修复）**：
1. 0.6B 模型 `text_hidden_size=2048` vs `hidden_size=1024`，`sft_12hz.py` 缺少 `text_projection` 导致维度不匹配 → 已补投影
2. checkpoint 的 `spk_id` 写成大写 `S06`，但模型用 `speaker.lower()` 查表 → 已统一小写
3. 微调模型采样时可能不终止（出现过 655s 音频）→ `demo.py` 加 `max_new_tokens=512` + `do_sample=False`

---

## 6. 模块说明

| 文件 | 功能 |
|------|------|
| `ranAudio/voice_clone/config.py` | 路径常量、超参数配置 |
| `ranAudio/voice_clone/data_prep.py` | 数据解析、参考音频筛选、数据集划分 |
| `ranAudio/voice_clone/resample.py` | 音频重采样 16kHz → 24kHz |
| `ranAudio/voice_clone/zeroshot.py` | 零样本 voice clone 推理 |
| `ranAudio/voice_clone/finetune.py` | 微调训练流程 |
| `ranAudio/voice_clone/evaluate.py` | 效果评估（Qwen3-ASR CER + cam++ 相似度） |
| `ranAudio/voice_clone/demo.py` | 推理 Demo CLI |
| `ranAudio/voice_clone/utils.py` | 工具函数 |

---

## 7. 关键发现与注意事项

1. **xvec 模式更适合构音障碍**：仅提取说话人向量，不迁移构音韵律/风格，生成更清晰
2. **音频采样率**：MDSD 数据为 16kHz，Qwen3-TTS 微调需 24kHz，需先重采样
3. **flash-attn 兼容性**：`sft_wrapper.py` 自动降级 `flash_attention_2` → `sdpa`
4. **参考音频选择**：选最长（>3s）的音频，speaker embedding 质量更好
5. **ICL 模式较慢**：需编码参考音频 code，单条推理约 350s；xvec 模式约 5-12s
6. **微调已验证失败并弃用**：用构音语音做微调目标会让模型学到构音缺陷，即使 2e-6 低学习率也未能保住清晰度（详见 §5.4）。零样本 xvec 是最终方案

---

## 8. 后续扩展

- 扩展到其他 6 位说话人（S01/S04/S08/S09/S12/S20）
- 尝试语音增强（enhance/）预处理参考音频
- 多参考音频融合提升 speaker embedding 质量
- 流式推理支持（Qwen3-TTS 支持 streaming）
