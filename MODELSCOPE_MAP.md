# ModelScope 发布清单 ↔ GitHub 仓库映射

> 生成时间：2026-11-20　　账号：`chenhaha12315`
> 现状：ModelScope 上 **0 个模型仓库 / 1 个数据集仓库**（`wenet-speech-L-filter`，Tier-A 已发布）
> 目标：把「自己训练的、公开渠道下不到的」产物发到 ModelScope；其余留 OSS / 新机

---

## 0. 三条判定原则

1. **基座已在 ModelScope → 不传**。11 个基座模型逐一验证均存在，重复上传只是浪费。
2. **自己的派生权重 → 传 ModelScope 模型仓库**（单文件须 ≤ 50 GB）。
3. **`optimizer.pt` / 中间 step checkpoint → 不传**，留 OSS。一个 12 G 的 checkpoint 去掉
   `optimizer.pt`（7.6 G）只剩 `model.safetensors`（3.8 G），体积直接砍 68%。

---

## 1. 已经在 ModelScope 上的基座（**不要传**）

| 本地路径 | 体积 | ModelScope 已有仓库 | 下载量 |
|----------|------|---------------------|--------|
| `ranAudio/kimi/models/Kimi-Audio-7B-Instruct/` | 44 G | `moonshotai/Kimi-Audio-7B-Instruct` | 27,194 |
| `rmsaqa-code/models/Qwen3-Omni-30B-A3B-Instruct/` | 66 G | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 257,699 |
| `MiniCPM-o-Demo/models/MiniCPM-o-4_5/` | 19 G | `OpenBMB/MiniCPM-o-4_5` | 273,622 |
| `Step-Audio/Step-Audio-2-mini/` | 17 G | `stepfun-ai/Step-Audio-2-mini` | 21,369 |
| `Qwen3-ASR/Qwen3-ASR-1.7B/` | 4.4 G | `Qwen/Qwen3-ASR-1.7B` | 1,191,639 |
| `Qwen/Qwen3.5-0.8B/` `2B` `9B` | 25 G | `Qwen/Qwen3.5-0.8B` `-2B` `-9B` | 257万 / 139万 / 346万 |
| `Qwen3-TTS/Qwen3-TTS-12Hz-*` | 9.6 G | `Qwen/Qwen3-TTS-12Hz-*` | 13万–47万 |

> **合计约 185 G，全部跳过。** 本地保留只是为了不重新下载，不是待发布物。

---

## 2. GitHub 仓库 ↔ ModelScope 模型仓库 映射

| # | GitHub 仓库 | 本地资产 | 原始体积 | **已构建**（实测） | 建议 ModelScope 仓库 | 优先级 |
|---|-------------|----------|---------:|-----------:|----------------------|--------|
| 1 | `chen-ming-yang/Qwen3-ASR` | `Qwen3-ASR/qwen3-asr-*`（9 个实验，各取最终 ckpt） | 157 G | **34.89 GiB** | `chenhaha12315/qwen3-asr-ft-wenetspeech` | **P0** |
| 2 | `chen-ming-yang/Qwen` | `Qwen/Qwen3.5-0.8B-merged` + `-merged-v4` | 11.6 G | **2.86 GiB** | `chenhaha12315/qwen3.5-0.8b-merged` | **P0** |
| 3 | `chen-ming-yang/ranAudio` | `ranAudio/runs/s2_fc_nofc/` 6 个 LoRA + 3 个 projector | 6.6 G | **6.10 GiB** | `chenhaha12315/ranAudio-s2-lora` | **P0** |
| 4 | `chen-ming-yang/cdsd` | `ranAudio/cdsd/results/moe_spk*/` 10 个 final ckpt + dense_lora | 29 G | **7.81 GiB** | `chenhaha12315/cdsd-moe-lora` | P0 |
| 5 | `chen-ming-yang/speech-sec` | `speechsec/checkpoints/st/best.pt` | 925 M | **0.90 GiB** | `chenhaha12315/speechsec-st` | P0 |
| 6 | `chen-ming-yang/MeanSE` | `MeanSE/exp/meanflow/{ncsnpp,ncsnpp_dysar}` | 5.0 G | **4.93 GiB** | `chenhaha12315/MeanSE-meanflow` | P1 |
| 7 | `chen-ming-yang/ranAudio` | `ranAudio/runs/s15_asr/*/` 各 run 的 `*_best.pt` | 80 G | **5.02 GiB** | `chenhaha12315/ranAudio-s15-asr` | P1 |
| 8 | `chen-ming-yang/ranAudio` | `ranAudio/runs/s2_fc_nofc_full/*/{llm_final,projector_final,tokenizer}` | 66 G | **32.95 GiB** | `chenhaha12315/ranAudio-s2-full` | P2 |
| 9 | `chen-ming-yang/MeanFlowSE123` | `MeanFlowSE123/MeanFLowSE/lightning_logs/` | 44 G | **❌ 不传** | — | 丢弃 |
| 10 | （无对应仓库） | `MiniCPM-o-Demo/models/MiniCPM-o-4_5-fc*`（5 个） | 90 G | 视需要 | `chenhaha12315/minicpm-o-fc` | P2 |
| 11 | `chen-ming-yang/FunASR` / `real-asr` | 仅代码，无自训权重 | — | — | — | 无需 |

> 第 1~8 项已全部制作完成，合计 **95.47 GiB / 344 文件**，位于 `/cmy/publish_ms/`，
> 详见该目录下 `REPORT.md` 与 `MANIFEST.tsv`。

> 第 9 项**不要传**：`MeanFlowSE123` 的 checkpoint 文件名是 `epoch=0_si_sdr=-60.37.ckpt`、
> `epoch=3_pesq=-inf.ckpt`，训练指标是 `-inf`，说明那次训练失败了，权重无价值。

### 2.1 说明：为什么「可发布体积」远小于「原始体积」

以 `Qwen3-ASR/qwen3-asr-finetune-again-out-20260512` 为例：

```
checkpoint-5000/
├── model.safetensors   3.8 G   ← 发布
├── optimizer.pt        7.6 G   ← 不发布（只有继续训练才需要）
├── trainer_state.json  ~20 K   ← 可选
└── ...
```

15 个 checkpoint 全部带 `optimizer.pt`，所以 157 G → 58.1 G（全发）或 34.2 G（每实验只发 final）。
每个 `model.safetensors` 均为 **3.88 G**，`optimizer.pt` 为 **7.6 G**，砍掉后体积减少约 **66%**。

### 2.2 制作发布包

```bash
# 以 Qwen3-ASR 某个实验为例：只挑 model.safetensors + tokenizer + config
SRC=/cmy/cmy/Qwen3-ASR/qwen3-asr-finetune-again-out-20260512/checkpoint-5636
DST=/tmp/publish/qwen3-asr-ft
mkdir -p "$DST"
cp "$SRC"/model.safetensors "$DST"/
cp "$SRC"/config.json "$SRC"/generation_config.json "$DST"/ 2>/dev/null
cp -r "$SRC"/tokenizer* "$SRC"/vocab* "$SRC"/merges* "$DST"/ 2>/dev/null
# 不要 cp optimizer.pt / trainer_state.json / scheduler.pt
du -sh "$DST"
```

---

## 3. 数据集发布映射

| # | 本地资产 | 体积 | 来源 | 建议 ModelScope 仓库 | 可行性 |
|---|----------|-----:|------|----------------------|--------|
| 1 | `wenetSpeechL/tier_A_10shards` 等 | 3.7 G | WenetSpeech + 三方一致筛选 | `chenhaha12315/wenet-speech-L-filter` | ✅ **已发布** |
| 2 | `Qwen3-ASR/data/augmented/` | 20 G | 自建增强数据 | `chenhaha12315/qwen3-asr-augmented` | ⚠️ 需确认底层音频许可 |
| 3 | `real-asr/data/manifests/{aishell,dysarthric,wenetspeech}` | 4.5 G | 自建 manifest | `chenhaha12315/asr-manifests` | ✅ 纯标注，最安全 |
| 4 | `ranAudio/data/` | 1.3 G | 自建 | 并入 `ranAudio-*` | ✅ |
| 5 | `rmsaqa-code/data/` | **342 G** | RMS-AQA 派生 | — | ❌ 超社区平台合理范围，留 OSS |
| 6 | `speechsec/data/` | 94 G | LibriTTS-R 派生特征 | — | ❌ 可重算，留本地 |
| 7 | `WenetSpeech/` | 2.1 T | 官方公开集 | — | ❌ 官方已有，重下 |

> 数据集的规律：**纯标注（JSONL/manifest）几乎必然可发；含音频的要先查底层授权。**

---

## 4. 建议执行顺序

| 步骤 | 动作 | 体积 | 预计耗时 |
|------|------|-----:|----------|
| 1 | 建 `qwen3-asr-ft-wenetspeech`，9 个实验各发 final ckpt | 34.2 G | ~50 min |
| 2 | 建 `ranAudio-s2-lora`，6 个 adapter | 6.6 G | ~10 min |
| 3 | 建 `qwen3.5-0.8b-merged`，2 个 merged | 11.6 G | ~18 min |
| 4 | 建 `speechsec-st`，best.pt 一格 | 0.9 G | ~2 min |
| 5 | 建 `cdsd-moe-lora`，10 个 run 各取 final ckpt | 10.2 G | ~16 min |
| 6 | 建 `MeanSE-meanflow` | 5.0 G | ~8 min |
| 7 | 建 `ranAudio-s15-asr`，只发 `*_best.pt` | 5.1 G | ~8 min |
| 8 | 建 `asr-manifests` 数据集 | 4.5 G | ~7 min |

> 前 5 步合计约 **63.5 G**，覆盖了最有价值的部分。按 OSS 实测 ~11 MB/s，总计约 98 分钟。
> 8 个模型仓库全部发完约 **92.6 G**，约 2.4 小时。

---

## 5. 许可注意事项（2026-11-20 已补齐）

| 项目 | 许可证 | 依据 / 说明 |
|------|--------|-------------|
| `Qwen3-ASR` | **Apache-2.0** | 仓库原有 `LICENSE` |
| `MeanSE` | **Apache-2.0** ✅ 新增 | `meanse/ncsnpp.py` 是 Google Research Apache-2.0 代码；基于 ESPnet（Apache-2.0）→ 必须兼容 Apache |
| `real-asr` | **Apache-2.0** ✅ 新增 | `model/icefall_*.py` 3 个文件是 Xiaomi icefall（Apache-2.0）→ 必须随附 Apache 条款 |
| `cdsd` | **Apache-2.0** ✅ 新增 | 适配器基于 Qwen3（Apache-2.0），保持同族 |
| `speechsec` | **MIT** ✅ 新增 | 论文非官方实现，代码独立编写，无上游拷贝 |
| `ranAudio` | **Apache-2.0** ✅ 新增 | 自有代码；含第三方子目录，另附 `THIRD_PARTY_NOTICES.md` |
| `MeanFlowSE123` | MIT（原有） | ✅ |
| `Qwen/Qwen3.5-0.8B-merged` | 目录内自带 `LICENSE` | ✅ 发布时随 `merged/`、`merged-v4/` 一起带上 |
| `MiniCPM-o-Demo` | 本地无 LICENSE | ⚠️ 基座属 OpenBMB，派生模型须遵守其模型协议（暂不发布） |
| Tier-A 数据集 | CC-BY-4.0 | ✅ 已发布 |

已新增文件：

```
MeanSE/LICENSE                      Apache-2.0 全文 (11,343 B)
MeanSE/NOTICE                       Google Research / ESPnet / Urgent2026
real-asr/LICENSE                    Apache-2.0 全文
real-asr/NOTICE                     icefall (Xiaomi) / Qwen3-ASR / k2
ranAudio/LICENSE                    Apache-2.0 全文
ranAudio/THIRD_PARTY_NOTICES.md     Kimi-Audio / CosyVoice / BigVGAN / Whisper / GLM-4-Voice
ranAudio/cdsd/LICENSE               Apache-2.0 全文
ranAudio/cdsd/NOTICE                Qwen3 / Transformers
speechsec/LICENSE                   MIT 全文
```

各仓库 README 也补了 `## License` 段落。

### 5.1 ⚠️ 一个必须注意的限制：GLM-4-Voice

`ranAudio` 内嵌了 GLM-4-Voice 的代码与 tokenizer 权重，它使用**自定义协议**（非 MIT/Apache）：

- 学术研究：免费
- **商业使用：需先在 https://open.bigmodel.cn/mla/form 登记**
- 明确「不可再许可（non-sublicensable）」

涉及路径：

```
ranAudio/kimi/kimia_infer/models/tokenizer/glm4/
ranAudio/kimi/models/glm-4-voice-tokenizer/
```

**这些子目录不得以 Apache-2.0 再发布。** 若要公开 `ranAudio`，建议在发布包中排除这两处，
让使用者自行下载；已在 `THIRD_PARTY_NOTICES.md` 中详细说明。

其余第三方均兼容：Kimi-Audio（Apache-2.0 + MIT）、CosyVoice（Apache-2.0）、
BigVGAN（MIT）、Whisper（Apache-2.0）。

---

## 6. 一键发布脚本模板

```bash
# 通用：发布一个已整理好的目录到 ModelScope 模型仓库
export MODELSCOPE_TOKEN=<你的 token>

publish_model() {
  local dir="$1" repo="$2" msg="${3:-upload}"
  python3 - "$dir" "$repo" "$msg" <<'PY'
import os, sys
from modelscope.hub.api import HubApi
d, repo, msg = sys.argv[1:4]
api = HubApi(); api.login(os.environ['MODELSCOPE_TOKEN'])
api.upload_folder(repo_id=repo, repo_type='model', folder_path=d, commit_message=msg)
print("DONE", repo)
PY
}
```

---

## 7. 一句话总结

- **不用传**：8 类基座模型（约 185 G，ModelScope 全都有）
- **值得传**：8 个模型仓库（约 92.6 G；前 5 个精选约 63.5 G）+ 2 个数据集（约 8 G）
- **别传**：`MeanFlowSE123` 的 44 G 失败训练日志、`rmsaqa-code/data` 342 G、`WenetSpeech` 2.1 T
- **对应关系**：GitHub 存**代码**，ModelScope 存**权重/数据**，同名或加后缀，一一对应

---

## 8. 相关文档

- 迁移总览与 OSS/ModelScope 通道对比：`迁移.md`（第 9 节）
- Tier-A 数据集（已发布）：https://www.modelscope.cn/datasets/chenhaha12315/wenet-speech-L-filter
- 发布脚本：`wenetSpeechL/push_tierA_to_modelscope.sh`
