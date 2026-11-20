---
license: apache-2.0
language:
  - zh
tasks:
  - automatic-speech-recognition
library_name: peft
tags:
  - moe
  - lora
  - personalized-asr
  - speaker-adaptation
  - qwen3
---

# cdsd · MoE 双视图个性化 ASR

在 Qwen3-ASR 上实现 **稠密 LoRA + MoE 专家** 的个性化中文 ASR。
每个说话人一套「共享稠密 LoRA + 专属 MoE 专家」，用于探索参数高效的多说话人适配。

## 核心结论

- 相对稠密基线（单 LoRA），MoE 方案 **CER 约降低 6 个百分点**
- **λ（路由/负载均衡权重）消融**：`λ=1` 最优
- **专家数消融**：`moe8` 优于 `moe16`（更多专家反而过拟合）

## 目录结构

```
moe_spk06/                       # 主实验（含完整消融）
├── dense_lora/                  # 稠密 LoRA 基线
│   └── adapter_model.safetensors        # ~20 MB
├── moe8_lam0/                   # 8 专家，λ=0
│   └── moe_state.pt                     # 750 MB
├── moe8_lam05/                  # 8 专家，λ=0.5
├── moe8_lam1/                   # 8 专家，λ=1  ← 最优
└── moe16_lam05/                 # 16 专家，λ=0.5（对照，850 MB）

moe_spk01/ moe_spk04/ moe_spk08/ moe_spk09/ moe_spk12/
└── dense_lora/ + moe8_lam1/     # 其它说话人，统一用最优配置

moe_spk_all/
└── moe8_lam1/                   # 全说话人混合训练
```

## 使用

`moe_state.pt` 是 MoE 专家权重，需配合本项目的 `infer_moe_1best.py` 加载；
`dense_lora/adapter_model.safetensors` 是标准 PEFT 格式，可直接用
`peft.PeftModel.from_pretrained` 挂到 Qwen3-ASR 基座上。

```bash
python cdsd/infer_moe_1best.py \
    --moe  moe_spk06/moe8_lam1/moe_state.pt \
    --lora moe_spk06/dense_lora \
    --audio test.wav
```

复跑评测：`bash cdsd/run_moe_cer_bg.sh`

## 训练数据

CDSD 中文语音数据集（多说话人），按说话人切分为个性化训练集。
音频不随仓库分发，请遵守 CDSD 原始数据使用协议。

## 许可证

Apache License 2.0。

- 适配器基于 Qwen3（Apache-2.0），许可证与其保持同族
- 见 `NOTICE`

## 相关仓库

- 代码：https://github.com/chen-ming-yang/cdsd
- 结论文档：仓库内 `MOE_TRAINING.md` 与 `results/moe_spk06/MoE_CER差距报告.md`
