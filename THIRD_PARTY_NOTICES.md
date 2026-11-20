# Third-Party Notices

`ranAudio` 本仓库自身代码以 **Apache License 2.0** 发布（见 `LICENSE`）。
但仓库中包含若干第三方组件，它们**不适用**本仓库的 Apache-2.0 条款，
各自保留原始许可证。分发或再使用时请遵守对应条款。

---

## 1. Kimi-Audio（`ranAudio/kimi/`）

来源：Moonshot AI — https://github.com/MoonshotAI/Kimi-Audio

其 README 声明：

> The model is based and modified from [Qwen 2.5-7B](https://github.com/QwenLM/Qwen2.5).
> Code derived from Qwen2.5-7B is licensed under the Apache 2.0 License.
> **Other parts of the code are licensed under the MIT License.**

因此 `ranAudio/kimi/` 下代码为 **Apache-2.0 + MIT** 混合。

## 2. CosyVoice（`ranAudio/kimi/kimia_infer/models/tokenizer/glm4/cosyvoice/`）

来源：Alibaba Inc. (authors: Xiang Lyu) / Mobvoi Inc. (Binbin Zhang)
文件头声明 **Apache License 2.0**。

> 注：该目录位于 `glm4/` 之下，但 `glm4/` 整体按 Apache-2.0 分发（见 §5.1），
> 可正常随仓库发布。

## 3. BigVGAN（`ranAudio/kimi/kimia_infer/models/detokenizer/vocoder/`）

来源：NVIDIA CORPORATION, 2024 — https://github.com/NVIDIA/BigVGAN
**MIT License**（adapted from jik876/hifi-gan, MIT）。

## 4. Whisper / Transformers（`ranAudio/kimi/kimia_infer/models/tokenizer/whisper_Lv3/`）

来源：The OpenAI Authors and The HuggingFace Inc. team
**Apache License 2.0**。

## 5. GLM-4-Voice（⚠️ 代码与权重许可不同，需分别对待）

**结论：代码可随仓库分发；权重不可，但权重本就不入库。**

### 5.1 代码 —— `ranAudio/kimi/kimia_infer/models/tokenizer/glm4/`

许可证：**Apache License 2.0**
署名：`Copyright 2024 GLM-4-Voice Model Team @ Zhipu AI`（见该目录下 `LICENSE`）

✅ 与仓库 Apache-2.0 兼容，可正常分发，保留原 LICENSE 与署名即可。
（仓库内 `kimi/kimia_infer/models/tokenizer/glm4/LICENSE` 已随代码入库。）

### 5.2 权重 —— `ranAudio/kimi/models/glm-4-voice-tokenizer/`（1.4 GB）

许可证：**「glm-4-voice License」自定义协议**，不是 MIT/Apache。要点：

| 条款 | 内容 |
|------|------|
| 学术研究 | 免费使用 |
| **商业使用** | **需先在 https://open.bigmodel.cn/mla/form 完成登记**，登记后免费 |
| 再许可 | 明确「不可再许可（non-sublicensable）」 |
| 管辖 | 中华人民共和国法律；争议提交北京市海淀区人民法院 |

**处理方式：该目录已被 `.gitignore` 第 22 行 `kimi/models/` 排除，不进入本仓库，
也不进入任何 GitHub / ModelScope 发布包。** 使用者需自行从智谱官方渠道下载并遵守其协议。

> 因此**不需要**为它做额外排除操作，仓库本身即不含该权重。

## 6. Qwen 系列（模型权重，不随本仓库分发）

`Qwen/`、`Qwen3-ASR/` 等权重目录不在本仓库版本控制内，其许可证随各自模型
（Qwen3、Qwen3-ASR 均为 **Apache-2.0**）。

---

## 汇总

| 组件 | 路径 | 许可证 | 是否兼容 Apache-2.0 发布 |
|------|------|--------|--------------------------|
| 本仓库自有代码 | 其余目录 | Apache-2.0 | — |
| Kimi-Audio | `kimi/` | Apache-2.0 + MIT | ✅ 兼容（保留声明） |
| CosyVoice | `kimi/.../glm4/cosyvoice/` | Apache-2.0 | ✅ |
| BigVGAN | `kimi/.../vocoder/` | MIT | ✅ |
| Whisper | `kimi/.../whisper_Lv3/` | Apache-2.0 | ✅ |
| GLM-4-Voice **代码** | `kimi/.../glm4/` | Apache-2.0 | ✅ 随仓库分发 |
| GLM-4-Voice **权重** | `kimi/models/glm-4-voice-tokenizer/` | 自定义 | ⛔ 不入库，使用者自行下载 |
