# CS2W 口语→书面数据 / 评测

数据来自 [guozishan/CS2W](https://github.com/guozishan/CS2W)（CC BY-SA 4.0）。

## 划分（硬约束）

| 文件 | 用途 |
|------|------|
| `train.jsonl` / `val.jsonl` | 扩容 + TTS；开发评测用 val |
| `test.jsonl` | **封存**；不扩容、不 TTS、不调参 |

## 扩容 + 多引擎 TTS（后台）

```bash
cd /cmy/cmy/ranAudio/kimi
# 全量后台：规则扩容 train/val → edge → qwen_custom → qwen_design
bash scripts/run_cs2w_expand_tts_bg.sh
tail -f eval_spoken2written/expand_tts.log

# 只要 edge：CS2W_TTS_ENGINES=edge bash scripts/run_cs2w_expand_tts_bg.sh
# GPU 批量（默认 16，可调）：CS2W_TTS_BATCH=16 bash scripts/run_cs2w_expand_tts_bg.sh
```

产物：

- `expanded/{train,val}.jsonl`、`manifest.json`
- `wavs/{edge,qwen_custom,qwen_design}/*.wav`（16 kHz）
- `expanded/tts_index.jsonl`

```bash
python scripts/eval_spoken2written.py              # 默认 val（规则）
python scripts/eval_spoken2written.py --split smoke
```

## 文字 LoRA SFT（口语→书面）

扩容文本就绪后即可训练（**不依赖 TTS**）。默认基座 `Qwen3.5-0.8B`，环境 `qwen3`。

```bash
cd /cmy/cmy/ranAudio/kimi
# 后台：等 expanded → 打 messages jsonl → LoRA SFT
bash scripts/run_sft_spoken2written_bg.sh
tail -f eval_spoken2written/sft.log

# 可选：等 TTS 跑完再训（默认不等）
# SFT_WAIT_TTS=1 bash scripts/run_sft_spoken2written_bg.sh
# 调参示例：SFT_EPOCHS=3 SFT_BATCH=4 SFT_LR=1e-4 bash scripts/run_sft_spoken2written_bg.sh
```

产物：

- `sft/{train,val}.jsonl`、`sft/manifest.json`（user: `口语转书面：\n{content}` → assistant: `ref`）
- `sft_out/qwen35_0.8b_lora/`（LoRA adapter）
