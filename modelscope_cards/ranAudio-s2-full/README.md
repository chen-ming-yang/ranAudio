---
license: apache-2.0
language:
  - zh
tasks:
  - audio-classification
  - automatic-speech-recognition
library_name: pytorch
tags:
  - audio-gating
  - minicpm
  - speech
  - routing
---

# ranAudio S2 · 音频门控全参微调（7 个分支）

MiniCPM5-2B 的 **音频门控（audio gating）** 全参数微调权重：给定一段语音，
模型判断应当 **转写为文本（ASR）** 还是 **直接对话回复（chat）**，
或把请求 **委派给更强的后端模型（DELEGATE）**。

## 分支与实测效果

音频侧编码器来自 S1.5（SenseVoice encoder + projector），本仓库提供 S2 阶段
（LLM 全参）各课程分支的最终权重：

| 分支 | 训练策略 | 路由准确率 | 听写 CER | 备注 |
|------|----------|-----------:|---------:|------|
| `route` | 只训 FC + BASE | — | — | 先钉住「漏外部能力」，召回优先 |
| `chat` | 只训 NOFC chat | — | — | 学会闲聊、禁止复读问句 |
| `asr` | 只训 NOFC asr | — | — | 听写触发走 NOFC，陷阱不执行 |
| `mix` | 三桶混合 | 0.963 | — | 防灾难遗忘 |
| `mix_boundary` | 边界数据 + asr_weight=1 | **1.000** | ~0.267 | 路由最佳 |
| `mix_proj` | 只训 projector | — | 0.262 | 仅作对照，收益很小 |
| `mix_llmcont` | 换 llm_cont 编码 + 冻编码训 mix | **1.000** | **0.259** | 综合最佳（体验页默认） |

> 说明：`mix_llmcont` 同时达到路由 1.0 与最低听写 CER 0.259，是当前推荐分支。
> `mix_proj` 说明「只训投影层」几乎无收益，可作为消融参考。

## 目录结构

```
<branch>/
├── llm_final.safetensors     # 4.69 GB，LLM 全参
├── projector_final.pt        # 音频 projector
└── tokenizer/                # 分词器
```

## 使用

需配合 ranAudio 的推理脚本与 S1.5 音频编码器：

```bash
# 1) 准备 S1.5 编码器（见 ranAudio-s15-asr 仓库）
#    runs/s15_asr/b_ms_cont/{encoder_best.pt, projector_best.pt}

# 2) 指定 S2 分支
python eval_s2.py --adapter runs/s2_fc_nofc_full/mix_llmcont
```

协议与训练细节见仓库 `docs/S2_FC_NOFC_训练计划.md`。

## 训练数据

WenetSpeech（问句/听写）、自建闲聊文本、函数调用 XML 模板、
听写触发词陷阱集。音频不随仓库分发。

## 许可证

Apache License 2.0。第三方组件声明见 `THIRD_PARTY_NOTICES.md`。

## 相关仓库

- 代码：https://github.com/chen-ming-yang/ranAudio
- LoRA 版本（体积小 5 倍）：`chenhaha12315/ranAudio-s2-lora`
- 音频编码器：`chenhaha12315/ranAudio-s15-asr`
