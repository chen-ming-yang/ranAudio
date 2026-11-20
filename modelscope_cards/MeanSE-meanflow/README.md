---
license: apache-2.0
language:
  - zh
  - en
tasks:
  - audio-to-audio
  - text-to-speech
library_name: pytorch
tags:
  - speech-enhancement
  - mean-flow
  - diffusion
  - generative
  - 1-nfe
---

# MeanSE · Mean Flows 生成式语音增强

**MeanSE: Efficient Generative Speech Enhancement with Mean Flows**
（投稿 ICASSP 2026，[arXiv:2509.21214](https://arxiv.org/abs/2509.21214)）
的官方实现训练产物。

## 方法要点

流匹配（flow matching）类生成模型通常需要多次函数评估（NFE）才能达到稳定效果，
计算开销大、1-NFE 性能差。MeanSE 建模 **平均速度场（average velocity field）**，
在 **单次 NFE** 下即可获得高质量增强结果，并显著优于 flow matching 基线的域外泛化能力。

## 仓库内容

| 目录 | 训练目标 | Checkpoint |
|------|----------|------------|
| `ncsnpp/version_0/checkpoints/` | 通用语音增强 | `best_epoch=05-step=190000-val_loss=0.111.ckpt`（1.23 GB）<br>`last.ckpt`（1.23 GB） |
| `ncsnpp_dysar/version_0/checkpoints/` | 构音障碍语音增强 | `best_epoch=21-step=200000-val_loss=0.377.ckpt`（1.23 GB）<br>`last.ckpt`（1.23 GB） |

- `ncsnpp`：基于 NCSN++ 骨干的 Mean Flow 模型，通用域
- `ncsnpp_dysar`：在构音障碍语音上训练，val_loss 更高属预期（该域本身更难）

> 推荐使用 `best_*` 而非 `last`，可节省一半体积。

## 使用

```bash
# 1-NFE 推理
python inference.py \
    --ckpt ncsnpp/version_0/checkpoints/best_epoch=05-step=190000-val_loss=0.111.ckpt \
    --nfe 1 \
    --input noisy.wav \
    --output enhanced.wav
```

`--nfe` 控制函数评估次数；增大可提升质量但线性增加耗时，
MeanSE 的设计目标就是在 `--nfe 1` 下仍保持稳定。

## 训练

```bash
pip install espnet
bash train_meanflow.sh
```

## 评价指标

训练日志中以 **val_loss** 为选择依据。感知指标（PESQ / STOI / DNSMOS）请按论文
实验设置自行复现——本项目未附评测集音频。

## 许可证

Apache License 2.0。见 `NOTICE`：

- `meanse/ncsnpp.py` 源自 **Google Research**（DDPM / score_sde），Apache-2.0
- 框架基于 **ESPnet**（Apache-2.0）与 **Urgent 2026 Challenge** 代码框架

## 引用

```bibtex
@article{meanse2025,
  title   = {MeanSE: Efficient Generative Speech Enhancement with Mean Flows},
  journal = {arXiv preprint arXiv:2509.21214},
  year    = {2025}
}
```

## 相关仓库

- 代码：https://github.com/chen-ming-yang/MeanSE
- 论文：https://arxiv.org/abs/2509.21214
