---
license: apache-2.0
language:
  - zh
  - en
tasks:
  - automatic-speech-recognition
library_name: transformers
tags:
  - asr
  - qwen3
  - wenetspeech
  - chinese
  - speech-recognition
datasets:
  - chenhaha12315/wenet-speech-L-filter
---

# Qwen3-ASR 中文微调合集 · WenetSpeech-L

基于 **Qwen3-ASR-1.7B** 在中文语音数据上做监督微调的 9 组产物集合，用于中文 ASR
复现与对比实验。

## 仓库内容

| 目录 | 说明 |
|------|------|
| `finetune-again-20260512/checkpoint-5636` | 二次微调（2026-05-12） |
| `finetuning-all-out-20260524/checkpoint-3948` | 全量数据微调（2026-05-24） |
| `finetuning-all-out-20260616/checkpoint-3963` | 全量数据微调（2026-06-16，较新） |
| `finetuning-best/checkpoint-10432` | 训练步数最多的最佳档 |
| `finetuning-out-20260509/checkpoint-10432` | 首轮微调（2026-05-09） |
| `finetuning-out-20260524/checkpoint-340` | 小步数微调（2026-05-24） |
| `finetuning-out-20260615/checkpoint-230` | 小步数微调（2026-06-15） |
| `finetuning-out-bak/checkpoint-326` | 备份档 |
| `gspo-final/checkpoint-300` | GSPO 强化学习后训练 |

每个目录均为可直接加载的完整模型：

```
config.json
generation_config.json
model.safetensors          # 3.80 GB（gspo-final 为 4.39 GB）
preprocessor_config.json
tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
chat_template.jinja / chat_template.json
special_tokens_map.json / added_tokens.json
```

> 已剔除 `optimizer.pt`、`scheduler.pt`、`rng_state.pth`、`trainer_state.json`
> 等训练中间态，仅保留推理所需文件。

## 快速使用

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch, librosa

path = "finetuning-best/checkpoint-10432"
processor = AutoProcessor.from_pretrained(path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

audio, _ = librosa.load("test.wav", sr=16000)
inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

## 训练数据

- WenetSpeech-L（经三模型一致性筛选的子集，见
  [Tier-A 数据集](https://www.modelscope.cn/datasets/chenhaha12315/wenet-speech-L-filter)）
- AISHELL 系列
- 构音障碍（dysarthric）语音数据

> 训练音频**不随本仓库分发**，请自行获取原始语料。

## 适用与限制

- ✅ 中文普通话 ASR、领域微调基线、伪标签生成
- ⚠️ 训练数据以朗读/播客为主，对强噪声、远场、方言的鲁棒性未做系统评测
- ⚠️ 各 checkpoint 的 CER 未附在仓库内，请以实际评测为准

## 许可证

Apache License 2.0。

- `LICENSE` 为 **Qwen3-ASR 上游原文**（`Copyright 2026 Alibaba Cloud`），保持未修改
- `NOTICE` 说明本 fork 的改动范围（微调脚本、数据增强、评测工具、发布的权重归本仓库作者）

## 相关仓库

- 代码：https://github.com/chen-ming-yang/Qwen3-ASR
- 基座：https://github.com/QwenLM/Qwen3-ASR
