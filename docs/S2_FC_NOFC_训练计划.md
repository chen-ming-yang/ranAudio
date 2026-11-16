# S2 训练计划：三路由 `NOFC` / `FC+LOCAL` / `DELEGATE`

**状态**：协议 Draft **v0.3**（文档）· 旧 `mix_boundary` 实测见 §8（**旧 FC+BASE 协议**）· **日期**：2026-11-09  
**架构标准**：[MiniCPM5-2B_音频门控架构定义.md](./MiniCPM5-2B_音频门控架构定义.md) Draft v0.3 §3.2 / §10  
**实现对照**：[architecture.md](./architecture.md)  
**底座 XML**：[minicpm5_fc_format.md](./minicpm5_fc_format.md)

> **相对旧 S2**：废除「一律 BASE 上抛 / 主干吐转写」。旁路 ASR 始终开；FastLM 三路由。  
> **代码与已训权重未跟**：`protocol.py` / `mix_boundary` 仍为 FC+BASE+NOFC。§8 / §9 数字仅对旧协议有效。三路由造数与训练**另开实现**，本轮只改文档。

本阶段（v0.3 目标）结束时：FastLM 能用首 token 区分 **对话 / 本地工具 / 委派**；简单槽位本地出 function XML；复杂能力 DELEGATE，上派文本来自旁路 ASR。S1.5 的 Wenet 清单 **不改**。  
产品是**通用语音助手**，不限于车机。  
词表保留现有 token，**新增 `<|DELEGATE|>`**，启用 `<|LOCAL|>`；`<backend>` / `</backend>` 仍为回注输入标记（§2.1）。

---

## 1. 2B 输出分类（Draft v0.3 唯一字面）

2B 每轮只走下面三类（听写归 NOFC）。按真实输出分类。

| 类 | 整段输出 | 用户侧 | L2 |
|---|---|---|---|
| 对话 · 闲聊 | NOFC + 回答 | 你是谁、讲个笑话、一加一等于几 | 本地，送 TTS |
| 对话 · 听写 | NOFC + 正文 | 请转写 / 写成文字 + 正文 | 本地；剥掉听写指令 |
| 本地工具 | FC + LOCAL + `<function>` XML | 打开蓝牙、调高音量等简单车机槽位 | 本地解析执行 |
| 委派 | DELEGATE | 查天气/多跳/复杂规划/需强模型 | eos 上派；`asr_text`=旁路 ASR |

```text
<|NOFC|>我是语音助手。闲聊和问答我可以直接答。
<|NOFC|>今天会议取消了
<|FC|><|LOCAL|><function name="enable_bluetooth">
</function>
<|DELEGATE|>
```

- 首 token：`<|NOFC|>` / `<|FC|>` / `<|DELEGATE|>`。
- `<|FC|>` 后本版第二 token **固定 `<|LOCAL|>`**，再跟 MiniCPM5 风格 `<function>` XML（见 [minicpm5_fc_format.md](./minicpm5_fc_format.md)）。不要 `<|BASE|>` 转写上派。
- `<|DELEGATE|>` 本版默认**不跟转写正文**；上派字段见架构定义 §4.3。垫话标 **OPEN**。
- `<|NOFC|>` 后直接跟中文：闲聊是回答（禁止复读），听写是要记下的正文。
- 听写是对话不是工具：`翻译成文字：打开蓝牙` → `<|NOFC|>打开蓝牙`。触发词见 §3。
- **不回吐** `<backend>`（只在 user 输入，见 §2.1）。
- **历史**：旧协议 `FC+BASE+转写` 仅见于已训权重与 §8/§9；三路由不再依赖 BASE 上派。

---

## 2. 词表、system、L2

词表（v0.3 目标；代码未加 DELEGATE）：

`<|FC|>` `<|NOFC|>` `<|LOCAL|>` `<|BASE|>` `<|DELEGATE|>` `<backend>` `</backend>`

- `<|BASE|>`：历史兼容，三路由主训不用。
- `<|LOCAL|>`：启用；SFT **默认不传 `tools=`**，靠 system / 示例教 XML。
- `<|DELEGATE|>`：**新增**；eos 即 fire-and-forget。

字面用尖括号对。整段进 special tokens。现网常量仍见 [`protocol.py`](../ranAudio/protocol.py) 的 `S2_SPECIAL_TOKENS`（**无 DELEGATE**，待实现）。

jsonl（目标）：`route=nofc` → `kind=chat|asr`；`route=local` → function XML；`route=delegate` → 仅 DELEGATE。旧 `venue=base` 行仅兼容集。

建议 system（`tools=None`；目标字面，代码 `S2_SYSTEM` 未改）：

```text
你是语音助手。先输出路由标记，再输出对应内容，不要输出其它标记。

自己能答时：
<|NOFC|>理解结果
- 闲聊、问答：直接回答，不要复述用户的话
- 用户要转写/听写：只输出要记下的正文，去掉听写指令

简单车机槽位、端侧可执行时：
<|FC|><|LOCAL|><function name="工具名">
<parameter name="参数名">值</parameter>
</function>
（无参工具可只有 function 开闭标签。不要先转写。）

需要强推理、多跳、复杂规划或端侧无法可靠填槽时：
<|DELEGATE|>
（不要跟转写正文；转写由旁路 ASR 提供。）

不要输出其它标记、JSON。不要回吐 `<backend>`。
```

L2：  
- NOFC → 送 TTS。  
- FC+LOCAL → 解析 XML，本地执行。  
- DELEGATE → eos 发 `route.dispatch`，附旁路 ASR `asr_text`。  
- 误吐旧 BASE 转写：实现期可降级兼容，不得当作 v0.3 主路径。  
- 2B 误吐 `<backend>` 则剥掉再送 TTS。

### 2.1 异步回注输入（`<backend>`）

`<backend>` 是 **user 侧输入标记**，不是 2B 输出协议。载荷 = SlowLM / Slave 的 `speak` / `confirmation`，不要旁路 ASR。无回注时不要空标签，只放音频括号。

同一轮 user 串：

```text
<backend>北京明天多云转晴，18到24度。</backend><|audio_start|>…pads…<|audio_end|>
```

2B 任务：

- **不**回吐 `<backend>` / `</backend>`。
- 标签内事实 **原样** 出现在输出里（数字、专名、温度区间不得改写）。
- 再和当前音频 Cond 的那路输出拼成 **一段** 给 TTS 的话：可加衔接（「查到了」「对，」），可接上用户这秒说的话；不得用生成先验替换 backend 事实。
- 字面输出：`<|NOFC|>` + 这段中文。有回注的这轮是播报，不再上抛。无新命令、只有回注时也走 NOFC。

```text
in:  <backend>北京明天多云转晴，18到24度。</backend> + 音频「嗯，温度呢」
out: <|NOFC|>查到了，北京明天多云转晴，18到24度。
```

禁止：「18到24度」→「二十度左右」；丢掉数字；输出标签字面。验收：backend 中的数字必须出现在 2B 输出（实体/数字保真）。

本版 jsonl **不造** 这条。词表先加齐；「backend + 音频 → 拼接 TTS」另开桶。

组串已接线：[`build_user_content`](../ranAudio/model.py)；`generate_text(wav, backend=speak)` / `teacher_force_batch(..., backend=)`。`from_pretrained` 用 `CHAT_SPECIAL_TOKENS` 一次加入六个 S2 token。

---

## 3. 听写触发（归 NOFC，触发词是一族）

用户不一定说「请转写」。同一句里**有正文**才造数：剥掉指令，只留正文。

触发词（造数时轮换，可再加近义，不要锁死这一张表）：

请转写、帮我转写、转写成文字、写成文字、转成文字、帮我记一下、记下来、翻译成文字、把我说的话写成字

正文可以跟在冒号、逗号、空格后面，或直接接上。

| 用户 | 标签 |
|---|---|
| 请转写：今天会议取消了 | NOFC + 今天会议取消了 |
| 写成文字，明天三点开会 | NOFC + 明天三点开会 |
| 翻译成文字：打开蓝牙 | NOFC + 打开蓝牙（要的是字，不是开蓝牙） |
| 打开蓝牙 | FC + BASE + 打开蓝牙 |
| 把这句话翻译成英语 | NOFC + 英译（闲聊，不是听写） |

「翻译成文字 / 转成文字」= 听写。「翻译成英文 / 英语」= 闲聊。

光杆（「转写我说的话」「翻译下我说的话为文字」）产品上应 ack「好的，请说要转写的内容」。本版 jsonl **不造**光杆，也不做下一轮历史。不进入听写模式（OPEN-4）。

---

## 4. 数据

**S1.5 不改。** `data/wenet_s.jsonl` 仍是纯转写 + `ASR_SYSTEM`。标签里加路由会让冻结的 2B 被 encoder 推向「凡是语音先出 FC」。详见 [S1.5_SenseVoice_训练计划.md](./S1.5_SenseVoice_训练计划.md)。

S2 另造 jsonl：

| 桶 | 来源 | 目标 |
|---|---|---|
| FC + BASE | 车机 22 + [`s2_toolish_texts.jsonl`](../data/s2_toolish_texts.jsonl)（天气/时间/搜索等从原闲聊拆出） | `<|FC|><|BASE|>` + 用户原话 |
| NOFC chat | [`chat_texts.jsonl`](../data/chat_texts.jsonl)（2B 能答）+ Wenet 问句预览 | `<|NOFC|>` + 回答（禁止先转写再回答） |
| NOFC asr | 听写触发族 + 同一句正文；不灌 Wenet | `<|NOFC|>` + 正文 |
| FC + LOCAL | — | 本版空着 |
| NOFC + backend | 回注 `speak` + 当前 wav | 本版空着。以后：`<|NOFC|>` 拼接稿；事实原样 |

禁止：听写标 FC；外部能力标 NOFC；Wenet 整表打成 FC；一条 wav 同时监督转写+闲聊；复用 `data/judge_audio.jsonl`（那边两边都有 asr）；S2 本版混入 backend 行。Wenet 闲聊预览另见 [`data/s2_wenet_chat_preview.jsonl`](../data/s2_wenet_chat_preview.jsonl)，不改 `wenet_s.jsonl`。

课程见 **§6.1**：四阶段接力，不是三个独立模型，也不是一上来混三桶。盯漏外部能力、NOFC 复读、BASE 后 CER。不测 LOCAL。

天气、时间、股价、搜索、路况本版算 **FC**。filler 是否由协调层出仍待定（不挡本版）。

---

## 5. 评测

- **该命令**：首 token `<|FC|>`，第二 `<|BASE|>`，再 CER + 实体/数字保真
- **该理解 chat**：`<|NOFC|>` + 非复读；**不算** CER
- **该理解 asr**：`<|NOFC|>` + CER
- 吐出 `<|LOCAL|>` / `<function>`：失败
- 吐出 `<backend>` 字面：失败（L2 可剥，训练仍记失败）
- 以后 backend 桶：`<|NOFC|>` + 非复读改写；backend 数字/专名保真 = 1.0

召回优先：漏命令（该 FC 却 NOFC）代价最高。

---

## 6. 开训配方（本版）

S1.5 已收官。S2 **只读**这对权重，不要混 v1.0 Whisper projector / LoRA：

```text
runs/s15_asr/b_ms_cont/encoder_best.pt
runs/s15_asr/b_ms_cont/projector_best.pt
```

| 模块 | 本版 |
|---|---|
| SenseVoice encoder | **冻** |
| Projector | **冻**（转写已沉在 S1.5；现网 `train_s2_route.py` 仍 `requires_grad=True`，改脚本时关掉） |
| MiniCPM5-2B | **LoRA，不全参**。现网草稿：`r=16`，`alpha=32`，`dropout=0.05`，`target_modules=q_proj,v_proj`，`lr=2e-4` |
| `tools=` | **不传** |

Loss（定义 §10.2，落到本版 token）：

| 段 | 权重 |
|---|---|
| 路由 token（FC / NOFC / BASE） | ≥ 1.0 |
| BASE 后转写 | 0.1–0.3（现网 `--asr-weight` 默认 0.2） |
| NOFC 后正文 | 1.0 |
| backend 输入 | 不算 assistant，无 loss |

### 6.1 四阶段课程（本版）

同一套 LoRA，四个阶段**接力**（后一阶段 `--resume` 前一阶段 ckpt）。不要每个桶单独训完丢弃，也不要一上来把三桶随机打乱。

| 阶段 | `--phase` | 数据 | 目的 |
|---|---|---|---|
| 1 | `route` | 只 FC+BASE | 先钉住漏外部能力（召回优先） |
| 2 | `chat` | 只 NOFC chat | 学会闲聊回答、禁止复读 |
| 3 | `asr` | 只 NOFC asr | 听写触发走 NOFC；陷阱不执行 |
| 4 | `mix` | 三桶一起 | 防灾难遗忘；chat 可 `neg_repeat` |

mix 里 FC 仍应略多于听写。2B 本版 LoRA；若加大 `r` / `target_modules` 后闲聊仍复读或 T2T 崩，再考虑全参，且必须加语音-文本交错回放。RL（路由 DPO）归 S3。

`assistant_token_weights` 开训前要认 `<|FC|>` / `<|BASE|>` / `<|NOFC|>`（现网还没改）。

**先改脚本再造数。**

| 文件 | 现状 | 开训前 |
|---|---|---|
| [`build_route_samples.py`](../ranAudio/build_route_samples.py) | 标签还不是 §1 | 改成 `format_fc_base` / `format_nofc`（[`protocol.py`](../ranAudio/protocol.py) 已有） |
| [`train_s2_route.py`](../ranAudio/train_s2_route.py) | `--phase` 只有三段；projector 可写 | `--phase` 改为 `route\|chat\|asr\|mix` + `--resume`；system 换成 §2；默认路径改 `b_ms_cont/*_best.pt`；冻 projector |
| [`eval_route.py`](../ranAudio/eval_route.py) | 解析还不是 §1 | 按 §5：首 token FC/NOFC，第二 BASE，再 CER / 保真 / 复读 |

词表：`from_pretrained` 已用 `CHAT_SPECIAL_TOKENS` 一次加齐六个 S2 token，造数不必再 `resize`。本版仍不造 LOCAL / backend 行。

---

## 7. 与 v0.3 的关系（旧「以后」已并入目标协议）

§1 已将 **LOCAL / DELEGATE** 升为目标字面；本节保留旧措辞供对照：

```text
# v0.3 目标
<|FC|><|LOCAL|><function name="enable_bluetooth"></function>
<|DELEGATE|>

# 旧 S2（已训权重）
<|FC|><|BASE|>帮我查北京天气再订餐厅
```

工具体复用 MiniCPM5 原生 XML。backend 拼接桶仍可另开，与三路由造数分开。

---

## 8. 边界续训（mix_boundary）

> **旧协议实测注记**：本节全部数字与标签针对 **FC+BASE+NOFC**（v0.2 / 代码现状），**不是** v0.3 三路由结果。路由 1.0 等指标不可直接外推到 LOCAL/DELEGATE。

四阶段全参停在 `runs/s2_fc_nofc_full/mix`（`<|im_end|>` 已计入损失）。测试集 `data/s2_test.jsonl`（82 条）上：route_acc 0.963，FC CER 0.115，ASR CER 0.280，不再复读。不再重跑 route / chat / asr，不解冻 encoder / projector。

剩下两类错：

- 路由边界 3 条。训练集有「背一首诗」「静夜思」，没有「春晓」；有「点外卖」「一杯咖啡」，没有「一份牛肉面」；没有「下棋」。
- 听写正文。路由是对的，但 `--asr-weight` 一直是 0.2，正文梯度弱，听不清时会编一句顺的。

这一轮从 mix 续训 **1 epoch**，只改数据和听写权重。

| 项 | 取值 |
|---|---|
| 起点 | `runs/s2_fc_nofc_full/mix`（`llm_final.safetensors` + tokenizer） |
| 输出 | `runs/s2_fc_nofc_full/mix_boundary` |
| 学习率 | `5e-6`（上一轮全参是 `1e-5`） |
| `--asr-weight` | `1.0`（路由 token 仍是 1.0；BASE 后转写和听写正文提到与路由相同） |
| `--boundary-repeat` | `8`（`source=boundary` 的行在课程里再重复；chat 仍乘 `--neg-repeat 2`） |
| encoder / projector | 继续冻。权重仍是 S1.5 `b_ms_cont/*_best.pt` |
| 脚本 | [`scripts/run_s2_boundary_mix.sh`](../scripts/run_s2_boundary_mix.sh) |

数据不改 `data/s2_audio.jsonl`。新行写到 `data/s2_boundary_jobs.jsonl`，edge-tts 合成到 `data/s2_boundary_wavs/`，再和原表拼成 `data/s2_audio_boundary.jsonl`。只用 edge，不占训练用的 GPU。

| 桶 | 条数 | 内容 | 标签 |
|---|---|---|---|
| chat | 62 | 春晓、登鹳雀楼、悯农、咏鹅、静夜思、绝句；下棋 / 象棋 / 围棋 / 五子棋；口算 | `<|NOFC|>` + 回答，禁止复读问句 |
| FC | 54 | 「下单 / 订餐」+ 具体菜名，含测试原句「帮我下单一份牛肉面」 | `<|FC|><|BASE|>` + 用户原话 |
| asr trap | 12 | 上面几句再套听写触发词 | `<|NOFC|>` + 正文，不执行 |

合计 128 条，全部走 edge。

三条已知漏判（「背一首春晓」「你会不会下棋」「帮我下单一份牛肉面」）写进这批，用来钉住边界；其余是近义说法。chat 不得命中 `needs_external` 的关键词，FC 必须命中「下单」或「订餐」。

评测：`eval_s2.py --adapter runs/s2_fc_nofc_full/mix_boundary`，结果 `tmp/s2_test_eval_full_mix_boundary.json`。日志 `tmp/s2_boundary_mix.log`。

结果（82 条，2026-09-23）：

| | mix（停重复之后） | mix_boundary |
|---|---|---|
| route_acc | 0.963 | **1.0** |
| FC CER | 0.115 | **0.046** |
| 闲聊路由 / 复读 | 0.875 / 0.125 | **1.0 / 0** |
| 听写 CER | 0.280 | 0.267 |
| 听写陷阱路由 | 1.0 | 1.0 |

「帮我下单一份牛肉面」走 `<|FC|><|BASE|>` 且转写全对。「背一首春晓」走 `<|NOFC|>` 并背出全诗。「你会不会下棋」路由对，回答漂到别的话题。听写字错误率几乎没动：地库 B3、氛围灯、沪深三百、厦门机票仍是近音或把听写指令留在正文里。把 `asr_weight` 抬到 1.0 没有把转写写进 2B。下一步见 §9，不再续训 LLM。

---

## 9. 投影层听写续训（mix_proj）

> **旧协议实测注记**：本节针对旧 S2 的 NOFC 听写 / BASE 后转写 CER，**不是** v0.3 旁路 ASR 指标。v0.3 上派转写改由旁路 ASR，本节结果仅作编码侧对照。

听写字错误率停在 0.267。S1.5 约定转写沉在 encoder + projector，不要再解冻 2B。这一轮只训投影层（`Linear(512→2048)→ReLU→Linear(2048→2048)`），编码器和 LLM 都冻住。

| 项 | 取值 |
|---|---|
| LLM | `runs/s2_fc_nofc_full/mix_boundary`，`--freeze-llm`，权重原样写入输出目录 |
| 投影层起点 | `runs/s15_asr/b_ms_cont/projector_best.pt`，`--train-projector` |
| encoder | 继续冻，`b_ms_cont/encoder_best.pt` |
| 数据 | `data/s2_audio_boundary.jsonl`，`--phase asr`（约 2510 条，含陷阱）。不进闲聊和 FC |
| 损失 | `--asr-weight 1.0`。梯度只进投影层 |
| 学习率 | 投影层 `1e-5`（S1.5 从随机初始化用 `1e-4`；这里已经对齐过） |
| epoch | 1 |
| 输出 | `runs/s2_fc_nofc_full/mix_proj`：`llm_final.safetensors` + tokenizer + `projector_final.pt` |
| 脚本 | [`scripts/run_s2_proj_asr.sh`](../scripts/run_s2_proj_asr.sh) |

评测必须带上新投影层，默认路径仍是 S1.5：

```text
eval_s2.py --adapter runs/s2_fc_nofc_full/mix_proj \
  --projector runs/s2_fc_nofc_full/mix_proj/projector_final.pt
```

结果写 `tmp/s2_test_eval_full_mix_proj.json`。日志 `tmp/s2_proj_asr.log`。

留下这份投影层的条件：路由仍 ≥ 0.98，听写字错误率明显低于 0.267。路由掉下去就退回 mix_boundary。听写指令没剥干净（「把我说的话写成字」留在正文）是 2B 的习惯，这一轮不指望它消失；近音才是投影层该动的部分。若字错误率仍在 0.27 附近，再考虑给 encoder 一个更小的学习率，仍然不训 2B。

结果（82 条，2026-09-23）：路由仍是 1.0，闲聊复读仍是 0，FC 字错误率 0.050（原先 0.046）。听写字错误率 0.262，没有明显低于 0.267。地库 B3、校门口原句没变；氛围灯从「分体灯」变成「范围灯」，沪深三百从「护身三百」变成「护眼三百」。**不替换** mix_boundary 使用的 S1.5 投影层。权重留在 `runs/s2_fc_nofc_full/mix_proj` 只作对照。

---

## 10. 融入 `b_ms_llm_cont` 听写（mix_llmcont）

背景：S1.5 解冻 2B 后续训到 `b_ms_llm_cont`（AISHELL-50 best CER **0.0179**）。S2 现网仍绑 `b_ms_cont` 编码侧 + `mix_boundary` 路由 2B。要把更好听写融进 S2，**保留** `mix_boundary` LLM（否则冲掉 FC/NOFC），只换 encoder/projector，再视需要续训。详见 [S1.5 §9.9](./S1.5_SenseVoice_训练计划.md)。

### 10.1 尝试方案总表（S2）

| 方案 | 目录 | 编码侧起点 | 训什么 | 代表结果 | 主路径？ |
|---|---|---|---|---|---|
| 四阶段全参 → mix | `runs/s2_fc_nofc_full/mix` | `b_ms_cont` | 2B 全参课程 | route ~0.963 | 否（boundary 起点） |
| mix_boundary | `…/mix_boundary` | `b_ms_cont` | 边界数据 + asr_weight=1 | route **1.0**；听写 CER ~0.267 | **当前 S2 LLM** |
| mix_proj | `…/mix_proj` | `b_ms_cont` | 只训 projector | 听写 CER 0.262，几乎无收益 | 仅对照 |
| C_swap | 无新 ckpt | **llm_cont** enc/proj + mix_boundary LLM | 零样本换编码 | S2 route 0.976；asr CER **0.334**（劣于 B） | 否 |
| D mix_llmcont | `…/mix_llmcont` | llm_cont enc/proj | resume boundary，冻 enc/proj，1 epoch mix | route **1.0**；asr CER **0.259**；fc CER 0.041 | **体验页默认（7860）** |

### 10.2 对照矩阵（量化「S2 吃掉多少听写」）

| 标签 | LLM | encoder/projector | 目的 |
|---|---|---|---|
| A_s15 | `b_ms_llm_cont/llm_best` | llm_cont best | 纯 S1.5 上限 |
| B_s2 | mix_boundary | `b_ms_cont` best | 现网 S2（实验前） |
| C_swap | mix_boundary | llm_cont best | 零样本换编码侧 |
| D_train | mix_llmcont | llm_cont best | 续训后 |

评测：

- 纯转写：`eval_asr.py --split aishell-test --max-samples 50` + `ASR_SYSTEM`（**只对 A 有意义**；路由 2B 在该 system 下仍会吐 `<|NOFC|>` / 改写腔，CER 不可比）
- 路由+听写：`eval_s2.py` + `data/s2_test.jsonl`（B/C/D 主表）

基线与训后（2026-09-27）：

| 标签 | AISHELL-50（ASR_SYSTEM） | S2 route_acc | S2 asr CER | S2 fc CER | chat parrot |
|---|---|---|---|---|---|
| A_s15 | **0.0179** | — | — | — | — |
| B_s2 | （不可比） | **1.0** | **0.267** | 0.046 | 0 |
| C_swap | （不可比） | 0.976 | **0.334** | 0.048 | 0.0625 |
| D_train | （不可比） | **1.0** | **0.259** | **0.041** | 0.0625 |

产物：`tmp/s15_asr_aishell50_{A_s15,B_s2,C_swap,mix_llmcont}.json`，`tmp/s2_test_eval_{B_s2,C_swap_llmcont,mix_llmcont}.json`。

读法：C 零样本换编码 **听写变差、路由略掉**；D 续训后路由回到 1.0，听写相对 B 略好（0.267→0.259），仍远差于纯转写 A（0.0179）——落差主要来自路由 2B / S2 任务，不是编码侧 alone。chat parrot 从 0 升到 0.0625，可接受。体验页默认切到 D。

### 10.3 训练配方（D）

脚本：[`scripts/run_s2_mix_llmcont.sh`](../scripts/run_s2_mix_llmcont.sh)

| 项 | 取值 |
|---|---|
| resume | `runs/s2_fc_nofc_full/mix_boundary` |
| encoder / projector | `runs/s15_asr/b_ms_llm_cont/*_best.pt`（**冻**） |
| 输出 | `runs/s2_fc_nofc_full/mix_llmcont` |
| phase / epoch | `mix` / 1 |
| lr / asr_weight | `5e-6` / `1.0` |
| 数据 | `data/s2_audio_boundary.jsonl`（不重做 TTS） |
| 日志 | `tmp/s2_mix_llmcont.log` |

**不要**用 `b_ms_llm_cont/llm_best.safetensors` 初始化 2B。训前停掉占卡 demo（7860 / 7861）。

成功标准：D 的 `route_acc` ≥ B；D 相对 B 听写 CER 下降；相对 A 的差距写入上表。若 C 已明显优于 B 且路由不掉，D 仍作对照；若 D 路由崩，保留 C。
