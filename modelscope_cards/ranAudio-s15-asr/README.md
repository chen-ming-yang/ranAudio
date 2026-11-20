---
license: apache-2.0
language:
  - zh
tasks:
  - automatic-speech-recognition
library_name: pytorch
tags:
  - sensevoice
  - zipformer
  - streaming-asr
  - dysarthria
  - speech
---

# ranAudio S1.5 · SenseVoice 编码器 8 组 ASR 变体

SenseVoice **encoder + projector** 的 8 组训练变体，作为 MiniCPM5-2B 音频门控
架构的听觉前端。每个变体只含编码侧权重，LLM 冻结。

## 变体与实测 CER

| 目录 | 训练配置 | AISHELL CER | 说明 |
|------|----------|------------:|------|
| `a/` | 只训 projector（冻 encoder + 2B） | ~0.193（前 50 条） | 热启基线；随机 projector 对照约 1.0（乱码） |
| `b_s/` | 解冻 encoder + projector | **0.0828**（best）/ 0.0978（收官） | S 数据 |
| `b_ms/` | MS 数据，冻 2B | 全量 ~0.0578 | 曾用作 S2 编码侧 |
| `b_ms_cont/` | `b_ms` 续训 | 全量 ~0.0578 / 前 50 条 **0.0316** | **S2 编码侧默认** |
| `b_ms_llm/` | 解冻 2B + enc + proj | 全量 ~0.0469 / 探测 **0.0232** | 续训起点 |
| `b_ms_llm_cont/` | 同上再 1 epoch | **0.0179**（best）/ 0.0209（收官） | **纯转写体验默认**，最佳 |
| `dysarthric/` | 构音障碍数据 e1 | — | encoder `3e-6`，projector `1e-5` |
| `dysarthric_e2/` | 从 e1 best 再热启 | — | e2 续训 |
| `smoke/` | 冒烟测试 | — | 仅用于跑通流程 |

> 结论：解冻 LLM（`b_ms_llm_cont`）明显优于只训编码侧；
> 但作为 S2 的听觉前端时，`b_ms_cont` 更稳（不冲掉上层 FC/NOFC 能力）。

## 目录结构

```
<variant>/
├── encoder_best.pt      # 0.82 GB，SenseVoice encoder
└── projector_best.pt    # 音频 → LLM 词表空间的投影层
```

`smoke/` 仅有 `projector_best.pt`。

## 使用

```python
import torch
enc = torch.load("b_ms_cont/encoder_best.pt", map_location="cpu")
proj = torch.load("b_ms_cont/projector_best.pt", map_location="cpu")
```

需配合 ranAudio 的 `runs/` 加载逻辑与 MiniCPM5-2B 基座。

## 训练数据

- AISHELL-1/2（主 CER 基准）
- WenetSpeech S / M / L
- 构音障碍（dysarthric）语音，针对发音不稳定、语速异常做数据增强

## 许可证

Apache License 2.0。第三方组件声明见 `THIRD_PARTY_NOTICES.md`。

## 相关仓库

- 代码：https://github.com/chen-ming-yang/ranAudio
- 上层门控：`chenhaha12315/ranAudio-s2-full`、`chenhaha12315/ranAudio-s2-lora`
