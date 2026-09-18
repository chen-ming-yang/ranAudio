# S2 训练计划：`<|FC|>` / `<|NOFC|>`（本版一律 BASE）

**状态**：协议已拍板 · **未开训** · **日期**：2026-09-13  
**架构标准**：[MiniCPM5-2B_音频门控架构定义.md](./MiniCPM5-2B_音频门控架构定义.md) §3.2 / §10  
**实现对照**：[architecture.md](./architecture.md)  
**底座 XML**：[minicpm5_fc_format.md](./minicpm5_fc_format.md)

本阶段结束时：FastLM 能用首 token 区分 **命令 vs 理解**；命令一律上抛 SlowLM。2B **不**出工具 XML。S1.5 的 Wenet 清单 **不改**。

---

## 1. 和定义 §4 的映射

定义写的是 `<mode:speak|route>` + `<asr>` / `<reply>` / `<filler>`。本版落地成更短的 special token，语义对齐，字面不同：

| 定义 | 本版 | 含义 |
|---|---|---|
| `<mode:route>` + `<asr>…</asr>` | `<|FC|><|BASE|>` + 转写 | 命令，上抛；eos ≈ `</asr>` 闭合 |
| `<mode:speak>` + `<reply>…</reply>` | `<|NOFC|>` + 理解正文 | 闲聊回答，或听写正文 |
| `<filler>` | 本版 2B **不出** | 垫话留给协调层（未定） |
| （定义未写本地执行） | `<|FC|><|LOCAL|>` + `<function>` | **只留 token，本版不训** |

`<|FC|>` = 这是命令，不是「必须上抛」。上抛还是本地由第二 token 决定。本版第二 token **固定 `<|BASE|>`**。简单命令以后在 2B 出 XML、复杂的继续上抛——两条会同时存在，本版先不做 LOCAL。

听写是 **理解**，归 NOFC，不是 FC。

---

## 2. 本版目标串

词表新增四个 special token（一次加齐，避免以后改 tokenizer）：

`<|FC|>` `<|NOFC|>` `<|LOCAL|>` `<|BASE|>`

监督只有两条：

```text
理解     <audio> → <|NOFC|>           → 闲聊 / 听写
命令     <audio> → <|FC|><|BASE|>     → 转写 → dispatch
（以后） <audio> → <|FC|><|LOCAL|>    → <function> XML   ← 不造、不训、不测
```

例子：

```text
<|FC|><|BASE|>打开蓝牙
<|FC|><|BASE|>导航去公司
<|NOFC|>今天天气不错，适合出门。
<|NOFC|>今天会议取消了
```

不要写成 `<|FC|>打开蓝牙`（缺第二标记），也不要写成 `<function name="enable_bluetooth">`。

jsonl：`route=fc` 时 `venue` 固定 `base`；`route=nofc` 时 `kind=chat|asr`。不要 `venue=local` 行。

S2 **不要** 传 `tools=`。一传，chat template 会逼模型直接出 `<function>`，路由 token 出不来。本版也不灌 XML。

建议 system（`tools=None`）：

```text
你是车机语音助手。先输出一个路由标记，再输出对应内容，不要输出其它标记。

需要执行命令（蓝牙、导航、空调、音量、车窗、电话、音乐、座椅加热、大灯等）时：
<|FC|><|BASE|>用户原话的转写
不要输出工具名、参数或 <function>。

不需要执行命令时：
<|NOFC|>理解结果
- 闲聊、问答：直接回答，不要复述用户的话
- 用户要转写/听写：只输出要记下的正文，去掉「请转写」「帮我记一下」等指令；即使正文听起来像命令也不要执行
- 只说了「请转写我说的话」、没有正文：回答「好的，请说要转写的内容」

不要先转写再回答。不要输出 <mode>、<asr>、<reply>、JSON。
```

L2：见 `<|FC|>` 后必须再读到 `<|BASE|>`，攒转到 eos 再 `route.dispatch`。吐出 `<|LOCAL|>` 或 `<function>` 当失败。

---

## 3. 「请转写」（已钉死，归 NOFC）

| 用户 | 标签 |
|---|---|
| 请转写：今天会议取消了 | `<|NOFC|>今天会议取消了`（剥指令前缀） |
| 请转写我说的话（光杆） | `<|NOFC|>好的，请说要转写的内容` |
| 请转写：打开蓝牙 | `<|NOFC|>打开蓝牙`（要的是字，不是开蓝牙） |
| 打开蓝牙 | `<|FC|><|BASE|>打开蓝牙` |

不进入听写模式（定义 OPEN-4，v0.1 不做）。下一轮用户说出内容再标 `kind=asr`。

---

## 4. 数据

**S1.5 不改。** `data/wenet_s.jsonl` 仍是纯转写 + `ASR_SYSTEM`。标签里加路由会让冻结的 2B 被 encoder 推向「凡是语音先出 FC」。详见 [S1.5_SenseVoice_训练计划.md](./S1.5_SenseVoice_训练计划.md)。

S2 另造 jsonl：

| 桶 | 来源 | 目标 |
|---|---|---|
| FC + BASE | 车机 22 + 同分布扩写 | `<|FC|><|BASE|>` + 用户原话 |
| NOFC chat | 2B 能答对的闲聊 / QA | `<|NOFC|>` + 回答（禁止先转写再回答） |
| NOFC asr | 显式听写句；**少量** Wenet | `<|NOFC|>` + 正文 |
| FC + LOCAL | — | 本版空着 |

禁止：听写标 FC；命令标 NOFC；Wenet 打成 FC；一条 wav 同时监督转写+闲聊；复用 `data/judge_audio.jsonl`（那边两边都有 asr）。

课程：先分后混。FC+BASE → NOFC chat → NOFC asr → 再混。盯漏命令、NOFC 复读、BASE 后 CER。不测 LOCAL。

仍待定（不挡本版）：天气 / 股价算不算命令；filler 是否由协调层出。

---

## 5. 评测

- **该命令**：首 token `<|FC|>`，第二 `<|BASE|>`，再 CER + 实体/数字保真
- **该理解 chat**：`<|NOFC|>` + 非复读；**不算** CER
- **该理解 asr**：`<|NOFC|>` + CER
- 吐出 `<|LOCAL|>` / `<function>`：失败

召回优先：漏命令（该 FC 却 NOFC）代价最高。

---

## 6. 以后（本版不做）

简单命令 2B 本地执行、复杂的继续上抛，同一模型里并存：

```text
<|FC|><|LOCAL|><function name="enable_bluetooth"></function>
<|FC|><|BASE|>帮我查北京天气再订餐厅
```

工具体复用 MiniCPM5 原生 XML（`<function` id=18）。到时再灌 `venue=local`，协议不用改。哪些工具算简单，本版不划名单。
