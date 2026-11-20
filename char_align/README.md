# 构音障碍字级强制对齐（S06 金标）

旁路任务：给定 `wav + 文本`，输出每字 `{char,start,end}`。不进 S2 / FastLM。

## 数据

来源：`/cmy/cmy/rmsaqa-code/同事新标注_06_part1_合并.tar`（人工 onset 点击）。

```bash
python char_align/prepare_gold.py \
  --tar /cmy/cmy/rmsaqa-code/同事新标注_06_part1_合并.tar \
  --out-dir data/char_align/s06_part1
```

| split | n |
|---|---|
| train | 1012 |
| dev | 128 |
| test | 130 |
| dropped (null time 等) | 31 |

区间：`start_i=onset_i`，`end_i=onset_{i+1}`，末字 `end=duration`。

## 模型

- 骨干：`real-asr` Zipformer2 + CTC（`exp/finetune_dysarthric/epoch_30.pt`）
- 损失：帧级 CE（字区间）+ 辅 CTC + onset 高斯 BCE
- 推理：按字级 SP id 做 CTC forced-align → onset

## 环境

```bash
source /cmy/cmy/miniconda3/etc/profile.d/conda.sh
conda activate qwen3-asr
cd /cmy/cmy/ranAudio
export PYTHONPATH=/cmy/cmy/real-asr:$PWD
export PYTHONUNBUFFERED=1
```

## 训练

GPU 空闲时：

```bash
bash char_align/wait_and_train.sh
# 或直接：
python char_align/train_align.py --out-dir runs/char_align/s06_part1
```

CPU 冒烟（已跑通，loss 下降）：

```bash
python char_align/train_align.py --device cpu --overfit-n 8 --epochs 3 \
  --batch-size 2 --num-workers 0 --out-dir runs/char_align/overfit8_cpu
```

## 评测

```bash
# WhisperX 基线
python char_align/eval_align.py --split test --baseline whisperx \
  --out-json tmp/char_align_wx_test.json

# 微调后
python char_align/eval_align.py --split test --baseline whisperx \
  --checkpoint runs/char_align/s06_part1/best.pt \
  --out-json tmp/char_align_test_final.json
```

### test 全量对比（已跑，130 句 / 749 字）

| 指标 | WhisperX（69/130） | Zipformer CTC FA（微调前） | Zipformer + onset 头 | 小模型·mel 输入 | **小模型·波形输入（最终）** |
|---|---|---|---|---|---|
| 覆盖 | 69/130 | 130/130 | 130/130 | 130/130 | **130/130** |
| onset MAE | 0.202 s | 0.167 s | 0.449 s | 0.130 s | **0.122 s** |
| onset median | 0.097 s | 0.104 s | 0.384 s | 0.043 s | **0.037 s** |
| ≤50ms | 32.7% | 27.1% | 13.8% | 53.4% | **55.9%** |
| ≤100ms | 51.7% | 47.7% | 25.5% | 66.4% | **70.0%** |
| ≤200ms | 67.9% | 74.5% | 38.9% | 78.4% | **81.2%** |

> 波形版数字见 `tmp/char_align_small_wave_test_final.json`，mel 版见 `tmp/char_align_small_test_final.json`。两者解码超参 `min_gap_s=0.35` 均在 dev 上选定。

## 互补性实验结论

- **mel 版 vs 波形版**：不互补（包含关系）。波形 ⊇ mel 信息（mel 是波形的确定性函数），波形前端已学到比固定 mel 更适合本任务的表示，融合权重 α 扫 0→1 单调劣化，纯波形（α=0）最优。
- **onset 检测 vs CTC 序列约束**：弱互补、不稳定。用 CTC forced-align 的 onset 做排序锚点，按相邻 onset 中点切区间、区间内取 onset 概率峰（`eval_fusion.py`）：

| split | 纯波形 onset | CTC FA | onset+CTC 融合 |
|---|---|---|---|
| test | 0.1218 s | 0.1673 s | **0.1151 s** ✅ |
| dev | 0.1630 s | 0.2057 s | 0.1645 s ❌（持平） |

  融合在 test 上略优（MAE −6.7ms、≤50ms 55.6→56.5%），但 dev 上无收益——收益依赖 CTC 锚点质量（dev 的 CTC 更差，0.206 vs test 0.167），锚点切错时反而把 onset 峰关进错误区间。**结论：onset 头已具备高精度+顺序能力，CTC 粗锚点的边际价值小且不稳定；纯波形 onset 更轻更稳，作为定版。**

## 小模型（最终方案）

Zipperformer 骨干下采样后帧移 80ms，且 onset 头用 σ=40ms 的单帧高斯监督、信号太稀疏，学不到精确定位（test MAE 0.449s）。改走「小模型从零训练」，**直接吃 10ms 帧率的输入**，不受 8× 下采样限制。

两版输入对比（参数量、seed、BiGRU+onset 头、训练设置完全一致，唯一区别是前端第一层）：

- **mel 版**：输入 80 维 log-mel（DSP 算好频谱），前端 `Conv1d(80→128,k5)`。
- **波形版（更优，最终）**：输入 16kHz 原始波形，前端 `Conv1d(1→128,k400,stride=160)` 让网络自己学频谱，同样降到 10ms 帧率（stride 160 采样点 = 10ms，与 mel hop 严格对齐）。

结论：**波形前端略优于固定 mel**（test MAE 0.122 vs 0.130；dev 0.163 vs 0.191），可学习的滤波器组学到了比固定梅尔基更适合「字 onset 检测」的表示。

- 架构：前端 Conv + LayerNorm + 2×BiGRU(128) + Linear(→1)，约 **628k 参数**。
- 损失：帧级 onset 高斯 BCE（σ=30ms）。
- 推理：sigmoid 概率 → 局部极大 → 贪心 NMS（`min_gap_s`）取前 N 峰（N=字数）。
- 早停：dev onset MAE（`patience=5`），本跑 early stop @ epoch 7。

```bash
python char_align/train_small_wave.py --out-dir runs/char_align/small_wave   # 训练（波形版）
python char_align/eval_small_wave.py --split test --min-gap-s 0.35            # 评测

python char_align/train_small.py --out-dir runs/char_align/small_onset       # 训练（mel 版，对照）
python char_align/eval_small.py --split test --min-gap-s 0.35                # 评测
```

产物：`runs/char_align/small_wave/best.pt`（波形版，最终）、`runs/char_align/small_onset/best.pt`（mel 版）

## 产物

- `data/char_align/s06_part1/{train,dev,test}.jsonl` + `Audio/06/`
- `runs/char_align/s06_part1/best.pt`（GPU 训完后）
- `tmp/char_align_wx_*.json`
