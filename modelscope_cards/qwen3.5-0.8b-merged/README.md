---
license: apache-2.0
language:
  - zh
  - en
tags:
  - qwen
  - merged
  - small-language-model
  - 0.8b
---

# Qwen3.5-0.8B · 微调后合并模型

Qwen3.5-0.8B 经微调后 **merge** 回完整权重的版本，提供两个迭代：

| 目录 | 说明 |
|------|------|
| `merged/` | 早期合并版本 |
| `merged-v4/` | **最终版本**，推荐使用 |

两者均为约 1.40 GB 的 `model.safetensors` + 完整配置与分词器，
目录内自带 `LICENSE`。

## 用途

作为端侧 / 低资源场景的轻量基座，用于：

- 音频门控架构中的快速 LLM（配合 ranAudio 的 S2 分支）
- 中文短文本生成、意图分类、文字规范（口语→书面转换）

## 使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "merged-v4"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="bfloat16", device_map="auto")
```

## 说明

- 官方 `Qwen/Qwen3.5-0.8B` 已具备完整文档与更高下载量，**如需通用能力请优先使用官方版**
- 本仓库仅保留本项目微调流水线产出的 merge 结果，用于复现实验

## 许可证

Apache License 2.0（随目录分发，见 `merged/LICENSE`、`merged-v4/LICENSE`）。

## 相关仓库

- https://github.com/chen-ming-yang/Qwen
- 官方基座：https://huggingface.co/Qwen
