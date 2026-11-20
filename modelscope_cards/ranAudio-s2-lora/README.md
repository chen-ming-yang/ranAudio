---
license: apache-2.0
language:
  - zh
tasks:
  - audio-classification
  - automatic-speech-recognition
library_name: peft
tags:
  - lora
  - audio-gating
  - minicpm
  - speech
---

# ranAudio S2 · 音频门控 LoRA 版

与 `ranAudio-s2-full` 同一架构与任务，但 LLM 侧改用 **LoRA** 而非全参微调，
体积从 32.95 GiB 降到 6.10 GiB，便于低成本部署与复现。

## 任务定义

给定一段语音，LLM 判断：

| 输出 | 含义 |
|------|------|
| `<\|NOFC\|>` + 正文 | 需要转写（ASR） |
| `<\|NOFC\|>` + 回答 | 直接对话（chat），禁止复读问句 |
| `<\|DELEGATE\|>` | 委派给更强后端，并附旁路 `asr_text` |

## 目录结构

```
asr/    chat/    route/
├── adapter_epoch1/  adapter_model.safetensors   # 1.01 GB
├── adapter_final/   adapter_model.safetensors   # 1.01 GB
└── projector_final.pt                           # 音频 projector
```

三个分支对应三套课程：`route`（先钉住外部能力召回）、`chat`（闲聊）、
`asr`（听写触发）。`adapter_final` 为推荐使用版本。

## 使用

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("<MiniCPM5-2B 基座>")
model = PeftModel.from_pretrained(base, "asr/adapter_final")
```

音频侧需配合 S1.5 编码器（`ranAudio-s15-asr` 仓库）与 `projector_final.pt`。

## 与全参版对比

| | 本仓库（LoRA） | `ranAudio-s2-full`（全参） |
|---|---|---|
| 体积 | 6.10 GiB | 32.95 GiB |
| 分支数 | 3（asr / chat / route） | 7（含 mix 系列） |
| 显存需求 | 低 | 高 |
| 路由 / CER | 略低 | `mix_llmcont` 最佳 |

## 许可证

Apache License 2.0。第三方组件声明见 `THIRD_PARTY_NOTICES.md`。

## 相关仓库

- 代码：https://github.com/chen-ming-yang/ranAudio
- 全参版本：`chenhaha12315/ranAudio-s2-full`
- 音频编码器：`chenhaha12315/ranAudio-s15-asr`
