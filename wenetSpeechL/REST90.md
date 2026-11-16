# WenetSpeech L 剩余 ~90% 一键流水线

与已完成的 **10% Tier-A** 相同流程：Qwen3-ASR-1.7B → FireRedASR2-AED → 三方一致筛选。  
使用与 10% **同一哈希划分** 的补集（`--fraction 0.1 --complement`），不与已跑 10% 重叠。

**当前状态（详见 [`TIER_A_RESULTS.md`](TIER_A_RESULTS.md)）**：清单、双模型推理、**`filter`（Tier-A）均已完成**；产物 `outputs/tier_A_agree3_rest90.jsonl` 共 **9,141,237 句 / 5,571.6 h**。末次 CER 约 Qwen **4.48%** / FireRed **3.00%**。

脚本支持断点续跑；筛三方一致只需：

```bash
bash launch_rest90.sh filter
```

---

## 一键开跑

```bash
cd /cmy/cmy/ranAudio/wenetSpeechL

# 后台：建补集清单（若无）→ Qwen → FireRed → Tier-A 筛选
bash launch_rest90.sh all
```

### 其它子命令

| 命令 | 作用 |
|------|------|
| `bash launch_rest90.sh prepare` | 只生成 `sample_manifest_rest90.jsonl` |
| `bash launch_rest90.sh qwen` | 只跑 Qwen（后台，可续跑） |
| `bash launch_rest90.sh firered` | 只跑 FireRed（后台，可续跑） |
| `bash launch_rest90.sh filter` | 只筛 Tier-A（后台；日志 `outputs/tier_A_filter_rest90.log`） |
| `bash launch_rest90.sh status` | 看进度 |
| `bash launch_rest90.sh stop` | 停任务（shard 保留，可再 `all` 续跑） |

### 监控

```bash
bash launch_rest90.sh status
tail -f outputs/pipeline_L90pct.log
tail -f outputs/qwen3_asr_L90pct/run.log
tail -f outputs/firered_asr_L90pct/run.log
wc -l outputs/qwen3_asr_L90pct/shard_*.jsonl
```

---

## 断点续跑

- 推理脚本读已有 `shard_*.jsonl` 里的 `utt_id`，**跳过已完成**。
- 异常/`stop` 后直接再执行：

```bash
bash launch_rest90.sh all
# 或分别
bash launch_rest90.sh qwen
bash launch_rest90.sh firered
bash launch_rest90.sh filter
```

- 清单 `sample_manifest_rest90.jsonl` 若已存在不会重建（保证划分稳定）。

---

## 输出路径

| 文件/目录 | 说明 |
|-----------|------|
| `sample_manifest_rest90.jsonl` | 剩余 ~90% 监督清单 |
| `outputs/qwen3_asr_L90pct/` | Qwen 结果 shards |
| `outputs/firered_asr_L90pct/` | FireRed 结果 shards |
| `outputs/tier_A_agree3_rest90.jsonl` | 本批 Tier-A |
| `outputs/tier_A_summary_rest90.json` | 统计 |
| `outputs/pipeline_L90pct.log` | 总流水线日志 |

与 10% 结果合并开源时，可拼接：

- `tier_A_open/data/tier_A_agree3.jsonl`（已有 10%）
- `outputs/tier_A_agree3_rest90.jsonl`（本批）

---

## 环境（与 10% 相同）

- Python：`/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python`
- Qwen：`/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B`
- FireRed：`/cmy/cmy/FireRedASR2S/pretrained_models/FireRedASR2-AED`

可用环境变量覆盖：`PYTHON` `QWEN_MODEL` `FIRERED_MODEL` `DEVICE` `QWEN_BS` `FIRERED_BS`。

---

## 实现说明

- `sample_L_manifest.py --fraction 0.1 --complement`：与 10% 相同 md5 分桶，取补集。  
- `run_qwen3_asr_compare.py` / `run_firered_asr_compare.py`：流式读 manifest，**不全量进内存**；按 shard 落盘。  
- `filter_tier_A.py`：官方 = Qwen = FireRed（去标点）。
