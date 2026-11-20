---
license: mit
language:
  - zh
  - en
tasks:
  - text-to-speech
  - audio-to-audio
library_name: pytorch
tags:
  - speech-synthesis
  - speech-editing
  - speech-continuation
  - conformer
  - semantic-tokens
---

# SpeechSEC · 语音合成 / 编辑 / 续写统一框架

> ⚠️ **本项目是 Interspeech 2025 论文的非官方实现**（unofficial implementation）。
> 与论文作者无关，代码为本仓库作者独立编写。
> 论文：[SpeechSEC: Unified Multi-Task Framework for Speech Synthesis, Editing, and Continuation](https://www.isca-archive.org/interspeech_2025/liang25e_interspeech.pdf)

## 支持的三个任务

| 任务 | 输入 | 输出 |
|------|------|------|
| **Synthesis（合成）** | semantic tokens | 完整语音 |
| **Editing（编辑）** | 语音 + 待替换区间 + 新内容的 tokens | 局部被替换后的语音 |
| **Continuation（续写）** | 前缀语音 | 自然延续的后续语音 |

三者共享 Conformer 骨干，通过任务前缀区分。

## 仓库内容

```
best.pt      0.90 GB   单任务最佳 checkpoint
train.log             训练日志（含 loss 曲线记录）
LICENSE               MIT
```

## 使用

```bash
python inference.py \
    --ckpt best.pt \
    --task synthesis \
    --tokens tokens.npy \
    --output out.wav
```

评测：`python evaluate.py --ckpt best.pt --task {synthesis,editing,continuation}`

## 前置依赖

需自备 semantic tokenizer / codec（如 SpeechTokenizer、CosyVoice tokenizer）
将音频转为 semantic tokens 后再输入本模型。这些组件**不随本仓库分发**。

## 训练数据

LibriTTS-R（合成/续写）、自建编辑对。音频不随仓库分发。

## 许可证

**MIT License**。本项目代码为独立编写，故采用宽松的 MIT；
论文本身及其作者可能存在的官方代码不受本许可证约束。

## 相关仓库

- 代码：https://github.com/chen-ming-yang/speech-sec
- 论文：https://www.isca-archive.org/interspeech_2025/liang25e_interspeech.pdf
